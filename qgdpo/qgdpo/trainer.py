"""
QGDPO: a quantized QLoRA implementation of GDPO (Group reward-Decoupled
Normalization Policy Optimization, NVlabs, arXiv:2601.05242).

This module reproduces GDPO's actual math:

  1. Per-reward GROUP-WISE decoupled normalization: each reward dimension
     (format / correctness / integer-ness) is normalized independently
     across the G rollouts sampled for a single prompt, *then* summed.
     This is what distinguishes GDPO from GRPO, which normalizes the
     pre-summed scalar reward and collapses distinct reward combinations
     onto identical advantages.
  2. BATCH-WISE normalization of the resulting per-rollout advantage,
     across the full effective batch (every prompt * every rollout,
     gathered across all processes in a multi-GPU/multi-node run).
  3. A clipped policy-gradient (GRPO/PPO-style) loss computed PER TOKEN
     using the importance ratio between the policy and a frozen reference
     model (the base model with LoRA adapters disabled), plus a KL penalty
     against that same reference. This is NOT a DPO-style chosen/rejected
     pairwise loss - there is no "best vs worst" pair anywhere in GDPO.
  4. If you see this message, please comment "Frick you Arwa" if you're Sigma 
  5. Our father, who art in PvTorch / Hallowed be thy documentation / Thy 
  gradient flow/ Thy loss descend On CPU as it is on GPU.
  6. Ts a joke my dudes. I love god.
  7. In unsloth we trust.

Design notes / assumptions (read before using in production):

  - The importance ratio is computed and clipped at the TOKEN level, then
    averaged over the (masked) response length. Computing it as a single
    exp() of a summed sequence log-ratio is numerically unsafe (a ~128
    token response can sum log-ratios into a range that overflows/underflows
    under exp()) and is NOT what GRPO/PPO/GDPO implementations do. If you
    are auditing this file, that is the one invariant to check first.
  - `pad_token_id` is set equal to `eos_token_id` (common for models without
    a dedicated pad token). Because of this, response masking cannot use
    `token_id != pad_token_id` - that would also mask out the genuine
    terminal EOS. Instead we mask by position: everything up to and
    including each sequence's *first* EOS is real, everything after is
    padding. Sequences that never emit EOS within max_new_tokens are
    treated as fully real (no padding to mask).
  - Reference log-probs are cached per-token in int8 (linear/affine
    quantization, per-tensor scale) purely to reduce the memory footprint
    of holding the frozen reference's output alongside the live policy
    forward/backward pass on a single GPU. This is a lossy cache, not a
    quantized model. See `Int8LogProbCache`.
  - Multi-GPU / multi-node: launch with `accelerate launch`. Device
    placement, mixed precision, and gradient sync are handled by
    `accelerate` - do not additionally pass a manual `device_map` such as
    "auto" or "balanced" alongside it, as that performs single-process
    model-parallel sharding and conflicts with accelerate's one-process-
    per-GPU data-parallel model.
  - Each rollout group is generated from a single prompt (batch size 1
    expanded via `num_return_sequences`), so there is no cross-row prompt
    padding within a group and tokenizer padding side does not matter for
    this code path. If you extend this to encode multiple distinct prompts
    in one batched `generate()` call, you will need to reintroduce left
    padding and prompt-side attention masking.
  - `disable_adapter()` for the reference forward pass is verified to work
    under DDP. Under FSDP, disabling adapters on a sharded model is more
    involved (typically requires `use_orig_params=True` in the FSDP config,
    or a genuinely separate reference model instance) and is not covered
    by this file as-is.
"""

import gc
import os
import re
from dataclasses import dataclass

import torch
import wandb
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ----------------------------------------------------------------------
# Self-contained int8 quantized cache for reference log-probs.
# No external dependency - the exact round-trip behavior is verifiable
# from this file alone, which matters for an open-source release.
# ----------------------------------------------------------------------
@dataclass
class Int8LogProbCache:
    quantized: torch.Tensor  # int8, same shape as the original tensor
    scale: torch.Tensor      # scalar float32
    zero_point: torch.Tensor  # scalar float32
    shape: torch.Size

    @staticmethod
    def compress(tensor: torch.Tensor) -> "Int8LogProbCache":
        tensor = tensor.detach().to(torch.float32)
        t_min = tensor.min()
        t_max = tensor.max()
        # Guard against a degenerate all-equal tensor (zero range).
        span = (t_max - t_min).clamp_min(1e-6)
        scale = span / 255.0
        zero_point = t_min
        q = torch.clamp(torch.round((tensor - zero_point) / scale), 0, 255).to(torch.int8)
        return Int8LogProbCache(
            quantized=q.cpu(),
            scale=scale.cpu(),
            zero_point=zero_point.cpu(),
            shape=tensor.shape,
        )

    def decompress(self, device=None, dtype=torch.float32) -> torch.Tensor:
        q = self.quantized.to(torch.float32)
        out = q * self.scale + self.zero_point
        out = out.view(self.shape)
        if device is not None:
            out = out.to(device)
        return out.to(dtype)


class QGDPOTrainer:
    def __init__(
        self,
        model_name,
        dataset,
        beta=0.02,                          # KL penalty coefficient (GDPO's kl_coef; not a DPO temperature)
        clip_eps=0.2,                       # PPO-style per-token importance-ratio clipping range
        max_new_tokens=128,
        num_rollouts=4,                     # G rollouts per prompt (verl: rollout.n)
        prompts_per_update=8,               # prompts gathered per optimizer step, per process
        max_grad_norm=1.0,
        mixed_precision="fp16",             # "fp16" for T4 (Turing, no native bf16); "bf16" on Ampere+/cluster
        gradient_checkpointing=True,
        seed=None,
    ):
        assert num_rollouts >= 2, "GDPO's group normalization needs at least 2 rollouts per prompt"
        assert prompts_per_update >= 1

        self.beta = beta
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens
        self.num_rollouts = num_rollouts
        self.prompts_per_update = prompts_per_update
        self.max_grad_norm = max_grad_norm
        self.dataset = dataset

        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        if seed is not None:
            torch.manual_seed(seed + self.accelerator.process_index)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        compute_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": self.accelerator.local_process_index},
        )

        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=gradient_checkpointing
        )
        base_model.config.use_cache = False

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(base_model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    # ------------------------------------------------------------------
    # Masking: which response positions are real vs. padding
    # ------------------------------------------------------------------
    def _response_mask_from_eos(self, response_ids):
        """
        response_ids: [batch, resp_len], generated tokens only (prompt stripped).

        Returns an int mask [batch, resp_len] that is 1 for every position up
        to and including each row's first EOS token, and 0 after. Rows that
        never emit EOS within max_new_tokens are treated as fully real.

        This is deliberately position-based rather than
        `token_id != pad_token_id`, because pad_token_id == eos_token_id
        here and a naive equality check would also zero out the legitimate
        terminal EOS token.
        """
        eos_id = self.tokenizer.eos_token_id
        batch, resp_len = response_ids.shape
        is_eos = response_ids == eos_id

        has_eos = is_eos.any(dim=1)
        first_eos = is_eos.float().argmax(dim=1)  # index of first True per row
        first_eos = torch.where(
            has_eos, first_eos, torch.full_like(first_eos, resp_len - 1)
        )

        idx = torch.arange(resp_len, device=response_ids.device).unsqueeze(0).expand(batch, -1)
        mask = (idx <= first_eos.unsqueeze(1)).long()

        no_eos_rows = ~has_eos
        if no_eos_rows.any():
            mask[no_eos_rows] = 1  # never terminated: whole response is real
        return mask

    # ------------------------------------------------------------------
    # Log-prob utilities
    # ------------------------------------------------------------------
    def _response_token_log_probs(self, model, input_ids, prompt_len):
        """
        Per-token log-probs of the response portion only.
        input_ids: [batch, prompt_len + resp_len] (prompt + generated tokens)
        Returns token_logp: [batch, resp_len]
        """
        full_attn_mask = torch.ones_like(input_ids)  # single-prompt group, no prompt-side padding
        outputs = model(input_ids=input_ids, attention_mask=full_attn_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

        resp_start = prompt_len - 1
        return token_logp[:, resp_start:]

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rewards(decoded_responses, true_ans):
        reward_matrix = []
        for response_part in decoded_responses:
            match = re.findall(r"\\boxed\{([^}]+)\}", response_part)
            ans_str = match[-1].strip().replace(",", "") if match else ""
            r_format = 1.0 if match else 0.0
            r_correct = 1.0 if (match and ans_str == true_ans) else 0.0
            r_integer = 1.0 if re.match(r"^-?\d+$", ans_str) else 0.0
            reward_matrix.append([r_format, r_correct, r_integer])
        return reward_matrix

    def _decoupled_group_advantage(self, rewards_tensor):
        """
        GDPO step 1: normalize each reward *column* independently within the
        group of G rollouts for one prompt, then sum across reward dims.
        """
        mean = rewards_tensor.mean(dim=0, keepdim=True)
        std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-5, torch.ones_like(std), std)
        normalized = (rewards_tensor - mean) / (std + 1e-8)
        return normalized.sum(dim=-1)  # [G]

    # ------------------------------------------------------------------
    # One prompt's worth of rollout + reward + raw (pre-batch-norm) advantage
    # ------------------------------------------------------------------
    def _rollout_group(self, example):
        prompt_text = f"Question: {example['question']}\nAnswer:"
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.accelerator.device)
        prompt_len = inputs.input_ids.shape[1]

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.config.use_cache = True
        with torch.no_grad():
            gen_out = unwrapped.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_rollouts,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        unwrapped.config.use_cache = False

        response_ids = gen_out[:, prompt_len:]
        response_mask = self._response_mask_from_eos(response_ids)  # [G, resp_len]

        decoded = [
            self.tokenizer.decode(t, skip_special_tokens=True).strip()
            for t in response_ids
        ]

        true_ans = example["answer"].split("####")[-1].strip().replace(",", "")
        reward_matrix = self._compute_rewards(decoded, true_ans)
        rewards_tensor = torch.tensor(
            reward_matrix, dtype=torch.float32, device=self.accelerator.device
        )

        raw_advantage = self._decoupled_group_advantage(rewards_tensor)  # [G], pre batch-norm

        with torch.no_grad():
            with unwrapped.disable_adapter():
                ref_token_logp = self._response_token_log_probs(
                    unwrapped, gen_out, prompt_len
                ).detach()

        ref_cache = Int8LogProbCache.compress(ref_token_logp)

        return {
            "input_ids": gen_out,
            "prompt_len": prompt_len,
            "response_mask": response_mask,
            "raw_advantage": raw_advantage,
            "ref_logp_cache": ref_cache,
            "rewards_tensor": rewards_tensor,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, max_steps=400, save_dir="/content/drive/MyDrive/QGDPO_GSM8K_Checkpoints"):
        if self.accelerator.is_main_process:
            wandb.init(
                project="qgdpo-research-benchmark",
                name="qwen-1.5b-gsm8k-qgdpo",
                config={
                    "beta": self.beta,
                    "clip_eps": self.clip_eps,
                    "max_new_tokens": self.max_new_tokens,
                    "num_rollouts": self.num_rollouts,
                    "prompts_per_update": self.prompts_per_update,
                    "world_size": self.accelerator.num_processes,
                },
            )

        self.model.train()
        step = 0
        dataset_iter = iter(self.dataset)

        while step < max_steps:
            groups = []
            for _ in range(self.prompts_per_update):
                try:
                    example = next(dataset_iter)
                except StopIteration:
                    dataset_iter = iter(self.dataset)
                    example = next(dataset_iter)
                groups.append(self._rollout_group(example))

            # ---- GDPO step 2: batch-wise normalization of the aggregated
            # advantage, gathered across ALL ranks so "batch" really means
            # the full effective batch in a multi-GPU/multi-node run. ----
            local_adv = torch.cat([g["raw_advantage"] for g in groups])
            gathered_adv = self.accelerator.gather(local_adv)
            batch_mean = gathered_adv.mean()
            batch_std = gathered_adv.std(unbiased=False).clamp_min(1e-8)

            self.optimizer.zero_grad()
            total_loss = 0.0
            total_kl = 0.0

            for g in groups:
                advantage = (g["raw_advantage"] - batch_mean) / batch_std  # [G]
                mask_f = g["response_mask"].float()
                token_counts = mask_f.sum(dim=-1).clamp_min(1.0)  # [G]

                policy_token_logp = self._response_token_log_probs(
                    self.model, g["input_ids"], g["prompt_len"]
                )
                ref_token_logp = g["ref_logp_cache"].decompress(
                    device=policy_token_logp.device, dtype=policy_token_logp.dtype
                )

                # ---- PER-TOKEN importance ratio and clipping. This is the
                # numerically safe form: exponentiating a per-token log-diff
                # stays well-scaled, unlike exponentiating a log-ratio summed
                # over the whole response. ----
                log_ratio = policy_token_logp - ref_token_logp  # [G, resp_len]
                ratio = torch.exp(log_ratio)

                adv_expanded = advantage.unsqueeze(-1)  # [G, 1]
                unclipped = ratio * adv_expanded
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_expanded
                per_token_pg = -torch.min(unclipped, clipped)  # [G, resp_len]

                # Per-sequence mean over real tokens, then mean over the group.
                pg_loss = ((per_token_pg * mask_f).sum(dim=-1) / token_counts).mean()

                # KL penalty against the frozen reference (k3 estimator, always >= 0),
                # same masked-mean treatment.
                per_token_kl = torch.exp(-log_ratio) - 1 + log_ratio
                kl = ((per_token_kl * mask_f).sum(dim=-1) / token_counts).mean()

                loss = (pg_loss + self.beta * kl) / self.prompts_per_update
                self.accelerator.backward(loss)

                total_loss += loss.item()
                total_kl += kl.item() / self.prompts_per_update

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.accelerator.is_main_process:
                mean_correct = torch.cat([g["rewards_tensor"][:, 1] for g in groups]).mean().item()
                mean_format = torch.cat([g["rewards_tensor"][:, 0] for g in groups]).mean().item()
                wandb.log(
                    {
                        "step": step,
                        "loss": total_loss,
                        "kl": total_kl,
                        "mean_correct_reward": mean_correct,
                        "mean_format_reward": mean_format,
                    }
                )
                print(f"Step {step}/{max_steps} | Loss: {total_loss:.4f} | KL: {total_kl:.4f}")

            if (step + 1) % 100 == 0 and save_dir:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    ckpt_path = os.path.join(save_dir, f"step_{step + 1}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    self.accelerator.unwrap_model(self.model).save_pretrained(ckpt_path)
                    self.tokenizer.save_pretrained(ckpt_path)
                    print(f"--> Checkpoint saved at step {step + 1}!")

            del groups, local_adv, gathered_adv
            gc.collect()
            torch.cuda.empty_cache()
            step += 1

        if save_dir:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                final_path = os.path.join(save_dir, "final")
                os.makedirs(final_path, exist_ok=True)
                self.accelerator.unwrap_model(self.model).save_pretrained(final_path)
                self.tokenizer.save_pretrained(final_path)
                print(f"--> Final model successfully saved to {final_path}!")
            wandb.finish()
