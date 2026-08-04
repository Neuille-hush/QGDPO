import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .quantization import QuantizedReferenceCache
from .losses import qgdpo_loss

class QGDPOTrainer:
    def __init__(self, model_name, dataset, beta=0.1, max_new_tokens=128, num_return_sequences=4):
        self.model_name = model_name
        self.dataset = dataset
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 4-bit QLoRA configuration to fit safely in Colab T4 VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(base_model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(base_model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def _compute_log_prob(self, prompt, response):
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        gather_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        return gather_log_probs.sum()

    def train(self, max_steps=10):
        self.model.train()
        step = 0
        steps = max_steps
        
        for example in self.dataset:
            if step >= max_steps:
                break
                
            prompt_text = f"Question: {example['question']}\nAnswer:"
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            decoded_outputs = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            true_ans = example['answer'].split("####")[-1].strip()
            
            reward_matrix = []
            for text in decoded_outputs:
                response_part = text[len(prompt_text):].strip()
                match = re.findall(r'\\boxed\{([^}]+)\}', response_part)
                
                r_format = 1.0 if match else 0.0
                r_correct = 1.0 if (match and match[-1].strip() == true_ans) else 0.0
                ans_str = match[-1].strip() if match else ""
                r_integer = 1.0 if ans_str.isdigit() else 0.0
                
                reward_matrix.append([r_format, r_correct, r_integer])

            rewards_tensor = torch.tensor(reward_matrix, dtype=torch.float32, device="cuda")

            # GDPO Decoupled Normalization with micro-jitter safeguard and tensor shape fix
            mean = rewards_tensor.mean(dim=0, keepdim=True)
            std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True)
            
            if torch.all(std < 1e-5):
                lengths = torch.tensor([len(text) for text in decoded_outputs], dtype=torch.float32, device="cuda").unsqueeze(1)
                rewards_tensor = rewards_tensor + (lengths / (lengths.max() + 1e-5)) * 1e-3
                mean = rewards_tensor.mean(dim=0, keepdim=True)
                std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True) + 1e-8
            else:
                std = std + 1e-8

            normalized_rewards = (rewards_tensor - mean) / std
            advantages = normalized_rewards.sum(dim=-1)

            best_idx = torch.argmax(advantages).item()
            worst_idx = torch.argmin(advantages).item()

            if best_idx != worst_idx:
                best_response = decoded_outputs[best_idx][len(prompt_text):].strip()
                worst_response = decoded_outputs[worst_idx][len(prompt_text):].strip()
                
                self.optimizer.zero_grad()
                
                policy_chosen = self._compute_log_prob(prompt_text, best_response)
                policy_rejected = self._compute_log_prob(prompt_text, worst_response)
                
                with torch.no_grad():
                    ref_logps_dummy = torch.tensor([policy_chosen.item(), policy_rejected.item()], device="cuda")
                    q_chosen_cache = QuantizedReferenceCache.compress(ref_logps_dummy)
                    q_rejected_cache = QuantizedReferenceCache.compress(ref_logps_dummy)
                
                loss = qgdpo_loss(
                    policy_chosen=policy_chosen,
                    policy_rejected=policy_rejected,
                    ref_chosen_q=q_chosen_cache,
                    ref_rejected_q=q_rejected_cache,
                    beta=self.beta
                )
                
                loss.backward()
                self.optimizer.step()
                
                print(f"Step {step}/{steps} | QGDPO Loss: {loss.item():.4f} (Advantage Gap: {(advantages[best_idx] - advantages[worst_idx]).item():.4f})")
            else:
                print(f"Step {step}/{steps} | Skipped (Identical indices)")
            
            torch.cuda.empty_cache()
            step += 1
