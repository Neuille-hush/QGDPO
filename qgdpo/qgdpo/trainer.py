import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from datasets import Dataset
from .losses import qgdpo_loss
from .quantization import QuantizedReferenceCache

class QGDPOTrainer:
    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        beta: float = 0.1,
        lr: float = 1e-5,
        max_new_tokens: int = 96,
        num_return_sequences: int = 4
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.beta = beta
        self.lr = lr
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Native 4-bit QLoRA and BitsAndBytes setup
        print(f"[QGDPO] Loading {model_name} with native 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
                self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0}  # Forces all layers onto GPU 0 safely for training
        )

        self.model = prepare_model_for_kbit_training(self.model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Native 8-bit AdamW optimizer to prevent VRAM spikes
        self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.lr)

    def _compute_log_prob(self, prompt_text: str, response_text: str):
        full_text = prompt_text + response_text
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=384).to("cuda")
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=384).to("cuda")
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        input_ids = inputs.input_ids[:, 1:]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        response_log_prob = token_log_probs[:, prompt_len - 1:].sum(dim=-1)
        return response_log_prob

    def train(self, max_steps: int = None):
        self.model.train()
        steps = max_steps if max_steps is not None else len(self.dataset)
        
        print(f"[QGDPO] Starting native training loop for {steps} steps...")
        for step in range(min(steps, len(self.dataset))):
            example = self.dataset[step]
            prompt_text = f"Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\n{example['question']}"
            
            inputs = self.tokenizer(
                prompt_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to("cuda")
            
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
            
            scored_completions = []
            for text in decoded_outputs:
                response_part = text[len(prompt_text):].strip()
                match = re.findall(r'\\boxed\{([^}]+)\}', response_part)
                score = 1.0 if (match and match[-1].strip() == true_ans) else (0.3 if match else 0.0)
                scored_completions.append((score, response_part))
                
            scored_completions.sort(key=lambda x: x[0], reverse=True)
            best_score, best_response = scored_completions[0]
            worst_score, worst_response = scored_completions[-1]
            
            if best_score > worst_score:
                self.optimizer.zero_grad()
                
                policy_chosen = self._compute_log_prob(prompt_text, best_response)
                policy_rejected = self._compute_log_prob(prompt_text, worst_response)
                
                with torch.no_grad():
                    ref_logps_dummy = torch.tensor([policy_chosen.item(), policy_rejected.item()], device="cuda")
                    q_cache = QuantizedReferenceCache.compress(ref_logps_dummy)
                
                loss = qgdpo_loss(
                    policy_chosen=policy_chosen,
                    policy_rejected=policy_rejected,
                    ref_chosen_q=q_cache,
                    ref_rejected_q=q_cache,
                    beta=self.beta
                )
                
                loss.backward()
                self.optimizer.step()
                
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    
                print(f"Step {step}/{steps} | QGDPO Loss: {loss.item():.4f} (Best: {best_score}, Worst: {worst_score})")
            else:
                print(f"Step {step}/{steps} | Skipped (No performance contrast)")
        
        print("[QGDPO] Training completed successfully!")
