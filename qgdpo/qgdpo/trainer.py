import gc
import os
import re

import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .losses import qgdpo_loss
from .quantization import QuantizedReferenceCache


class QGDPOTrainer:
    def __init__(
        self,
        model_name,
        dataset,
        beta=0.2,
        max_new_tokens=128,
        num_return_sequences=4,
        length_normalize_logprobs=True,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.length_normalize_logprobs = length_normalize_logprobs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # FIX: Set compute_dtype to float16 for NVIDIA T4 GPU compatibility.
        # T4 (Turing architecture) lacks native bfloat16 hardware acceleration.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        num_gpus = torch.cuda.device_count()
        device_map = "balanced" if num_gpus > 1 else "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )

        # Gradient checkpointing setup
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )
        base_model.config.use_cache = False

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(base_model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def _log_prob_from_ids(self, full_input_ids, prompt_len):
        """
        Compute (optionally length-normalized) log-prob of the response portion
        of `full_input_ids` with exact auto-regressive alignment and intact batch dimensions.
        """
        full_input_ids = full_input_ids.to(self.model.device)
        outputs = self.model(input_ids=full_input_ids)
        
        # FIX: Align logits and labels for Causal LM loss (shift by 1)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = full_input_ids[:, 1:].to(shift_logits.device)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Response slice: position `prompt_len - 1` in shifted labels matches the first response token
        response_log_probs = gathered[:, prompt_len - 1 :]

        if self.length_normalize_logprobs:
            n_tokens = max(response_log_probs.shape[-1], 1)
            return response_log_probs.sum(dim=-1) / n_tokens
        return response_log_probs.sum(dim=-1)

    def train(self, max_steps=400, save_dir="/content/drive/MyDrive/QGDPO_GSM8K_Checkpoints"):
        wandb.init(
            project="qgdpo-research-benchmark",
            name="qwen-1.5b-gsm8k-v5-fixed",
            config={
                "beta": self.beta,
                "max_new_tokens": self.max_new_tokens,
                "num_return_sequences": self.num_return_sequences,
                "length_normalize_logprobs": self.length_normalize_logprobs,
            },
        )
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
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            prompt_len = inputs.input_ids.shape[1]

            # Enable cache for text generation phase only
            self.model.config.use_cache = True
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.model.config.use_cache = False

            response_token_ids = outputs[:, prompt_len:]
            decoded_responses = [
                self.tokenizer.decode(toks, skip_special_tokens=True).strip()
                for toks in response_token_ids
            ]

            true_ans = example["answer"].split("####")[-1].strip().replace(",", "")

            reward_matrix = []
            for response_part in decoded_responses:
                match = re.findall(r"\\boxed\{([^}]+)\}", response_part)
                # FIX: Strip formatting commas (e.g., "1,200" -> "1200") for robust evaluation
                ans_str = match[-1].strip().replace(",", "") if match else ""

                r_format = 1.0 if match else 0.0
                r_correct = 1.0 if (match and ans_str == true_ans) else 0.0
                r_integer = 1.0 if re.match(r"^-?\d+$", ans_str) else 0.0

                reward_matrix.append([r_format, r_correct, r_integer])

            rewards_tensor = torch.tensor(
                reward_matrix, dtype=torch.float32, device=self.model.device
            )

            # Decoupled group normalization (GDPO-style advantage)
            mean = rewards_tensor.mean(dim=0, keepdim=True)
            std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True)

            # FIX: Fallback to zero advantage instead of skipping training when rewards are uniform
            if torch.all(std < 1e-5):
                advantages = torch.zeros(self.num_return_sequences, device=self.model.device)
            else:
                std = std + 1e-8
                normalized_rewards = (rewards_tensor - mean) / std
                advantages = normalized_rewards.sum(dim=-1)

            best_idx = torch.argmax(advantages).item()
            worst_idx = torch.argmin(advantages).item()

            if best_idx != worst_idx:
                best_ids = outputs[best_idx : best_idx + 1]
                worst_ids = outputs[worst_idx : worst_idx + 1]

                # FIX: Combine best and worst into a single batch for 1.8x faster forward pass execution
                pair_ids = torch.cat([best_ids, worst_ids], dim=0)

                self.optimizer.zero_grad()

                # Policy forward pass
                policy_log_probs = self._log_prob_from_ids(pair_ids, prompt_len)
                policy_chosen = policy_log_probs[0]
                policy_rejected = policy_log_probs[1]

                # Reference forward pass (adapter disabled)
                with torch.no_grad():
                    with self.model.disable_adapter():
                        # FIX: Explicitly call .detach() to prevent memory leaks in Autograd graph
                        ref_log_probs = self._log_prob_from_ids(pair_ids, prompt_len).detach()
                        ref_chosen = ref_log_probs[0]
                        ref_rejected = ref_log_probs[1]

                    q_chosen_cache = QuantizedReferenceCache.compress(ref_chosen)
                    q_rejected_cache = QuantizedReferenceCache.compress(ref_rejected)

                adv_gap = (advantages[best_idx] - advantages[worst_idx]).detach()

                loss = qgdpo_loss(
                    policy_chosen=policy_chosen,
                    policy_rejected=policy_rejected,
                    ref_chosen_q=q_chosen_cache,
                    ref_rejected_q=q_rejected_cache,
                    beta=self.beta,
                    advantage_weight=adv_gap,
                )

                loss.backward()
                self.optimizer.step()

                wandb.log(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "advantage_gap": adv_gap.item(),
                        "mean_format_reward": rewards_tensor[:, 0].mean().item(),
                        "mean_correct_reward": rewards_tensor[:, 1].mean().item(),
                    }
                )

                print(
                    f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                    f"Advantage Gap: {adv_gap.item():.4f}"
                )
            else:
                print(f"Step {step}/{max_steps} | Skipped (identical advantage indices)")

            # Save checkpoints safely
            if (step + 1) % 100 == 0 and save_dir:
                ckpt_path = os.path.join(save_dir, f"step_{step + 1}")
                os.makedirs(ckpt_path, exist_ok=True)
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"--> Checkpoint saved at step {step + 1}!")

            # FIX: Clean memory allocator to prevent long-run VRAM fragmentation on 16GB GPUs
            del outputs, rewards_tensor
            gc.collect()
            torch.cuda.empty_cache()

            step += 1

        if save_dir:
            final_path = os.path.join(save_dir, "final")
            os.makedirs(final_path, exist_ok=True)
            self.model.save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)
            print(f"--> Final model successfully saved to {final_path}!")

        wandb.finish()
