import torch
import re
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .quantization import QuantizedReferenceCache
from .losses import qgdpo_loss

class QGDPOTrainer:
    def __init__(self, model_name, dataset, beta=0.2, max_new_tokens=128, num_return_sequences=4):
        self.model_name = model_name
        self.dataset = dataset
        self.beta = beta  # Tightened KL penalty to prevent persona hallucination
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
        
        # ALIGNED: Learning rate set to 1e-6 to match the official NVIDIA/verl scripts exactly
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def _compute_log_prob(self, prompt, response):
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        gather_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        return gather_log_probs.sum()

    def train(self, max_steps=400, save_dir="/content/drive/MyDrive/QGDPO_GSM8K_Checkpoints"):
        # Initializing wandb with a v3 run name to track this new hyperparameter setup
        wandb.init(project="qgdpo-research-benchmark", name="qwen-1.5b-gsm8k-v3-aligned")
        self.model.train()
        step = 0
        dataset_iter = iter(self.dataset)
        
        while step < max_steps:
            try:
                example = next(dataset_iter)
            except StopIteration:
                dataset_iter = iter(self.dataset)
                example = next(dataset_iter)
                
            prompt_text = f"Question: {example['question']}\nAnswer:"
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1, # ALIGNED: Protects against infinite token loops
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

            mean = rewards_tensor.mean(dim=0, keepdim=True)
            std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True)
            
            # FIXED: Safely skip the step if there's no reward variance (prevents training on garbage ties)
            if torch.all(std < 1e-5):
                print(f"Step {step}/{max_steps} | Skipped (No reward variance - all responses failed/tied)")
                step += 1
                continue

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
                
                adv_gap = (advantages[best_idx] - advantages[worst_idx]).item()
                
                wandb.log({
                    "step": step,
                    "loss": loss.item(),
                    "advantage_gap": adv_gap,
                    "mean_format_reward": rewards_tensor[:, 0].mean().item(),
                    "mean_correct_reward": rewards_tensor[:, 1].mean().item()
                })
                
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Advantage Gap: {adv_gap:.4f}")
            else:
                print(f"Step {step}/{max_steps} | Skipped (Identical indices)")
            
            # AUTO-CHECKPOINT to Google Drive every 100 steps
            if (step + 1) % 100 == 0 and save_dir:
                ckpt_path = os.path.join(save_dir, f"step_{step+1}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"--> Checkpoint saved to Google Drive at step {step+1}!")
            
            torch.cuda.empty_cache()
            step += 1
            
        # Final save
        if save_dir:
            final_path = os.path.join(save_dir, "final")
            self.model.save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)
            print(f"--> Final model successfully saved to Google Drive at {final_path}!")
            
        wandb.finish()
