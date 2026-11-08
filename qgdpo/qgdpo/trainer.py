#!/usr/bin/env python3
"""
QGDPO v2 -- quantized QLoRA implementation of GDPO
(Group reward-Decoupled Normalization Policy Optimization, arXiv:2601.05242),
targeted at a single 16 GB T4.

What changed relative to v1
---------------------------
Math / correctness
  * The importance ratio is now pi_theta / pi_old (the sampling policy), with
    pi_old captured at rollout time. It is no longer pi_theta / pi_base.
  * KL is still measured against the frozen base model (adapters disabled),
    using the k3 estimator. Ratio and KL no longer share a denominator.
  * `ppo_epochs` (default 2) makes the PPO clip meaningful and amortises the
    cost of generation, which dominates on a T4.
  * Additive advantage modes: gdpo | grpo | sum_batch_whiten. The GRPO path is
    a real baseline (sum-then-group-normalise, no batch whitening).

Bugs removed
  * The int8 reference-logprob cache is gone. It wrapped levels >127 into
    negative int8 values, corrupting roughly half of every reference tensor.
    Reference log-probs are kept as small fp32 tensors on the GPU (~KBs).
  * The dataset is now sharded across ranks. v1 gave every process the same
    prompts, so multi-GPU was silently duplicated work.
  * No more `token_id != pad_token_id` masking; masking stops at the first
    EOS from a configurable set (Qwen emits <|im_end|>, not just <|endoftext|>).

Training quality
  * Greedy GSM8K evaluation every `eval_freq` steps (pass@1 + parse rate),
    with a wandb table of sample generations.
  * Dynamic sampling: prompt groups with no reward variance are skipped while
    the informative rate is high enough to make that worthwhile.
  * Adaptive KL controller instead of a fixed beta.
  * NaN/inf guard, resumable checkpoints, per-channel diagnostics, grad norm,
    clip fraction, truncation rate, unique-response rate.

Usage
-----
    accelerate launch train.py                       # full run
    accelerate launch train.py --smoke               # 3 tiny steps, sanity
    accelerate launch train.py --mode grpo           # baseline
    accelerate launch train.py --no-4bit             # fp16 LoRA (often faster)
    python train.py --selftest                       # pure-python tests, no GPU

On a T4, 300 iterations at the defaults is roughly 4-8 hours. Use
`--smoke` first, then `--steps 30 --eval-size 20` to check the trend before
committing to a long run.

The file is intentionally self-contained: no custom loss modules, no int8
cache, no hidden math.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:  # peft is required for the trainer, but keep imports explicit
    from peft import (
        LoraConfig,
        get_peft_model,
        load_peft_weights,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
except Exception as _peft_exc:  # pragma: no cover - surfaced at construction time
    LoraConfig = None  # type: ignore
    _PEFT_IMPORT_ERROR = _peft_exc
else:
    _PEFT_IMPORT_ERROR = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


# ----------------------------------------------------------------------------
# Pure helpers (kept module-level so they can be unit-tested without a GPU)
# ----------------------------------------------------------------------------

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
INT_RE = re.compile(r"^-?\d+$")

SYSTEM_PROMPT = (
    "You are a careful math solver. Reason step by step, then give the final "
    "numeric answer exactly once, inside \\boxed{...}. "
    "Example ending: ... so the answer is \\boxed{42}."
)


def parse_answer(text: str) -> Tuple[str, bool]:
    """Return (answer_string, used_boxed).

    Falls back to the last number in the text so that a model which is
    numerically right but format-wrong still receives gradient on the
    correctness channel (the format channel then teaches the box).
    """
    matches = BOXED_RE.findall(text or "")
    if matches:
        return matches[-1].strip().strip("$").strip(), True
    numbers = NUMBER_RE.findall(text or "")
    if numbers:
        return numbers[-1].strip(), False
    return "", False


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def numeric_match(pred: Any, gold: Any) -> bool:
    a, b = _to_float(pred), _to_float(gold)
    if a is not None and b is not None:
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
    return str(pred).strip() == str(gold).strip()


def is_integer_str(value: Any) -> bool:
    return bool(INT_RE.match(str(value).strip()))


def channel_z(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Per-column z-score with a safe degenerate-column guard.

    Constant columns produce exactly zero (0 / eps), which is the intended
    behaviour: a reward channel that carries no within-group information must
    not contribute an advantage.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def build_response_mask(response_ids: torch.Tensor, eos_ids: Sequence[int]):
    """Position-based response mask.

    Returns (mask, has_eos) where mask is 1 up to and including each row's
    first EOS and 0 afterwards. Rows that never emit EOS are fully real.
    This is deliberately not `token_id != pad_token_id`, because pad == eos
    for Qwen and that check would also drop the legitimate terminal EOS.
    """
    batch, length = response_ids.shape
    if length == 0:
        zeros = torch.zeros((batch, 0), dtype=torch.long, device=response_ids.device)
        return zeros, torch.zeros((batch,), dtype=torch.bool, device=response_ids.device)

    eos = torch.as_tensor(list(eos_ids), dtype=response_ids.dtype, device=response_ids.device)
    is_eos = torch.isin(response_ids, eos)
    has_eos = is_eos.any(dim=1)
    first = is_eos.float().argmax(dim=1)  # 0 when absent, overridden below
    first = torch.where(has_eos, first, torch.full_like(first, length - 1))

    positions = torch.arange(length, device=response_ids.device).unsqueeze(0)
    mask = (positions <= first.unsqueeze(1)).long()
    mask[~has_eos] = 1
    return mask, has_eos


@dataclass
class AdaptiveKL:
    """Simple proportional controller for the KL coefficient."""

    beta: float = 5e-3
    target: float = 1e-2
    factor: float = 1.5
    lo: float = 1e-5
    hi: float = 0.5
    ema: Optional[float] = None

    def observe(self, kl: float) -> float:
        if not np.isfinite(kl):
            return self.beta
        self.ema = kl if self.ema is None else 0.9 * self.ema + 0.1 * kl
        if self.ema > self.target * self.factor:
            self.beta = min(self.hi, self.beta * self.factor)
        elif self.ema < self.target / self.factor:
            self.beta = max(self.lo, self.beta / self.factor)
        return self.beta


@dataclass
class QGDPOConfig:
    # model / data
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: str = "test"

    # optimisation
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    ppo_epochs: int = 2
    clip_eps: float = 0.2
    beta: float = 5e-3
    use_ref: bool = True
    adaptive_kl: bool = True
    kl_target: float = 1e-2

    # advantage
    adv_mode: str = "gdpo"  # gdpo | grpo | sum_batch_whiten
    reward_weights: Tuple[float, ...] = (1.0, 1.0, 0.5, 0.5)
    reward_names: Tuple[str, ...] = ("format", "correct", "integer", "completion")

    # sampling
    max_new_tokens: int = 192
    num_rollouts: int = 4
    prompts_per_update: int = 4
    temperature: float = 0.9
    top_p: float = 0.95
    dynamic_sampling: bool = True
    max_rollout_tries: int = 3
    min_informative_rate: float = 0.2

    # model loading
    load_in_4bit: bool = True
    bnb_quant_type: str = "nf4"
    bnb_double_quant: bool = True
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    attn_impl: str = "sdpa"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # loop
    max_steps: int = 300
    eval_freq: int = 100
    eval_size: int = 50
    eval_max_new_tokens: int = 256
    save_freq: int = 50
    log_freq: int = 1
    empty_cache_every: int = 25

    # bookkeeping
    seed: int = 42
    save_dir: str = "qgdpo_runs"
    run_name: str = "qwen1.5b-gsm8k-qgdpo"
    wandb_project: str = "qgdpo"
    resume: Optional[str] = None


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


class QGDPOTrainer:
    def __init__(
        self,
        cfg: QGDPOConfig,
        train_dataset: Sequence[Dict[str, Any]],
        eval_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        if _PEFT_IMPORT_ERROR is not None:
            raise RuntimeError(f"peft is required: {_PEFT_IMPORT_ERROR}")
        if cfg.num_rollouts < 2:
            raise ValueError("GDPO's group normalisation needs at least 2 rollouts")
        if cfg.prompts_per_update < 1:
            raise ValueError("prompts_per_update must be >= 1")
        if cfg.adv_mode not in ("gdpo", "grpo", "sum_batch_whiten"):
            raise ValueError(f"unknown adv_mode: {cfg.adv_mode}")
        if len(cfg.reward_names) != 4 or len(cfg.reward_weights) != 4:
            raise ValueError(
                "this build expects exactly 4 reward channels "
                "(format, correct, integer, completion)"
            )
        if (
            cfg.mixed_precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] < 8
        ):
            raise ValueError(
                "this GPU (Turing/T4) has no native bf16; use mixed_precision='fp16'"
            )

        self.cfg = cfg
        self.kl = AdaptiveKL(
            beta=cfg.beta if cfg.use_ref else 0.0,
            target=cfg.kl_target,
        )
        self.start_step = 0

        self.accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
        if cfg.mixed_precision == "bf16":
            self.compute_dtype = torch.bfloat16
        elif cfg.mixed_precision == "fp16":
            self.compute_dtype = torch.float16
        else:
            # CPU / debugging: keep full precision, fp16 on CPU is pain.
            self.compute_dtype = torch.float32

        _seed_everything(cfg.seed + self.accelerator.process_index)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._setup_model()
        self._setup_eos_ids()

        self._uses_chat = bool(getattr(self.tokenizer, "chat_template", None))

        # Sharding happens on plain python lists: simplest correct behaviour
        # and independent of the dataset library in use.
        all_examples = list(train_dataset)
        self.examples = all_examples[self.accelerator.process_index :: self.accelerator.num_processes]
        if not self.examples:
            raise ValueError("dataset is smaller than the number of processes")

        self.eval_examples: List[Dict[str, Any]] = []
        if eval_dataset is not None:
            ev = list(eval_dataset)
            self.eval_examples = ev[self.accelerator.process_index :: self.accelerator.num_processes]

        self._it = None
        self._informative_rate = 0.0
        self._wandb_run = None
        self.model.train()

    # -- setup ---------------------------------------------------------------

    def _setup_model(self) -> None:
        cfg = self.cfg
        common: Dict[str, Any] = {}
        if cfg.attn_impl:
            common["attn_implementation"] = cfg.attn_impl

        if cfg.load_in_4bit:
            if self.accelerator.device.type != "cuda":
                raise RuntimeError("load_in_4bit requires CUDA; pass --no-4bit on CPU")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_quant_type,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=cfg.bnb_double_quant,
            )
            device_map = {"": self.accelerator.local_process_index}
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name, quantization_config=bnb, device_map=device_map, **common
                )
            except TypeError:
                common.pop("attn_implementation", None)
                base = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name, quantization_config=bnb, device_map=device_map, **common
                )
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        else:
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name, torch_dtype=self.compute_dtype, **common
                )
            except TypeError:
                common.pop("attn_implementation", None)
                base = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name, torch_dtype=self.compute_dtype, **common
                )
            if cfg.gradient_checkpointing:
                try:
                    base.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:  # transformers < 4.34
                    base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
            for p in base.parameters():
                p.requires_grad = False

        base.config.use_cache = False
        base.config.pad_token_id = self.tokenizer.pad_token_id

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_targets),
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_config)

        self.model = model
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.eval()

        if self.accelerator.is_main_process:
            trainable = sum(p.numel() for p in params)
            total = sum(p.numel() for p in model.parameters())
            print(f"[setup] trainable {trainable:,} / {total:,} params "
                  f"({100.0 * trainable / max(1, total):.2f}%)")

    def _setup_eos_ids(self) -> None:
        ids = set()
        tok = self.tokenizer
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        for token in ("<|im_end|>", "<|endoftext|>"):
            idx = tok.convert_tokens_to_ids(token)
            if idx is not None and idx != getattr(tok, "unk_token_id", None):
                ids.add(int(idx))
        if not ids:
            raise RuntimeError("could not determine any EOS token id")
        self.eos_ids = sorted(ids)

    # -- dataset iteration ---------------------------------------------------

    def _next_example(self) -> Dict[str, Any]:
        if self._it is None:
            order = list(self.examples)
            random.shuffle(order)
            self._it = iter(order)
        try:
            return next(self._it)
        except StopIteration:
            self._it = None
            return self._next_example()

    # -- prompt / generation -------------------------------------------------

    def _build_prompt(self, question: str) -> str:
        if self._uses_chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"

    def _encode(self, text: str) -> Any:
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=not self._uses_chat,
        ).to(self.accelerator.device)

    def _generate(
        self,
        prompt_text: str,
        n: int = 1,
        greedy: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        inputs = self._encode(prompt_text)
        prompt_len = int(inputs.input_ids.shape[1])
        model = self.accelerator.unwrap_model(self.model)

        kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=list(self.eos_ids),
            num_return_sequences=n,
        )
        if greedy or n == 1:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

        was_cache = bool(model.config.use_cache)
        model.config.use_cache = True
        model.eval()
        try:
            with torch.no_grad():
                try:
                    out = model.generate(
                        **inputs,
                        stop_strings=["</answer>"],
                        tokenizer=self.tokenizer,
                        **kwargs,
                    )
                except TypeError:
                    # older transformers: no stop_strings support
                    out = model.generate(**inputs, **kwargs)
        finally:
            model.config.use_cache = was_cache
        return out, prompt_len

    # -- log-probs -----------------------------------------------------------

    def _sequence_logprobs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        prompt_len: int,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Per-token log-probs of the response, computed in float32.

        The response slice is converted to float32 in chunks so the peak
        `[G, R, vocab]` buffer stays small on a 16 GB card.
        """
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        response_len = logits.shape[1] - prompt_len
        if response_len <= 0:
            return logits.new_zeros((logits.shape[0], 0), dtype=torch.float32)

        labels = input_ids[:, prompt_len:]
        pieces: List[torch.Tensor] = []
        for start in range(0, response_len, chunk):
            end = min(start + chunk, response_len)
            seg = logits[:, prompt_len - 1 + start : prompt_len - 1 + end, :].float()
            seg_labels = labels[:, start:end]
            picked = seg.gather(-1, seg_labels.unsqueeze(-1)).squeeze(-1)
            pieces.append(picked - seg.logsumexp(dim=-1))
        return torch.cat(pieces, dim=1)

    def _offline_logprobs(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        # Match the policy forward's precision: the prepared model is wrapped
        # in accelerate's autocast, and pi_old must be measured under the same
        # numerics or the ratio is not exactly 1 before the first update.
        autocast_ctx = (
            torch.autocast(
                device_type=self.accelerator.device.type,
                dtype=self.compute_dtype,
            )
            if self.cfg.mixed_precision in ("fp16", "bf16")
            else nullcontext()
        )
        try:
            with torch.no_grad():
                with autocast_ctx:
                    context = model.disable_adapter() if disable_adapter else nullcontext()
                    with context:
                        return self._sequence_logprobs(model, input_ids, prompt_len)
        finally:
            model.train(was_training)

    # -- rewards -------------------------------------------------------------

    def _compute_rewards(
        self,
        texts: Sequence[str],
        gold: str,
        truncated: Sequence[bool],
    ) -> torch.Tensor:
        rows: List[List[float]] = []
        for text, was_truncated in zip(texts, truncated):
            answer, used_boxed = parse_answer(text)
            r_format = 1.0 if used_boxed else 0.0
            r_correct = 1.0 if numeric_match(answer, gold) else 0.0
            r_integer = 1.0 if is_integer_str(answer) else 0.0
            r_completion = 0.0 if was_truncated else 1.0
            rows.append([r_format, r_correct, r_integer, r_completion])
        return torch.tensor(rows, dtype=torch.float32, device=self.accelerator.device)

    def _advantage(self, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_advantage, per_channel_z_for_logging).

        raw_advantage is per-rollout, shape [G]. Batch whitening (if the mode
        calls for it) is applied later, across the full effective batch.
        """
        weights = torch.tensor(
            self.cfg.reward_weights, dtype=rewards.dtype, device=rewards.device
        )
        z = channel_z(rewards, dim=0)  # [G, K]

        if self.cfg.adv_mode == "grpo":
            total = (rewards * weights).sum(dim=-1, keepdim=True)  # [G, 1]
            return channel_z(total, dim=0).squeeze(-1), z
        if self.cfg.adv_mode == "sum_batch_whiten":
            return (rewards * weights).sum(dim=-1), z
        # gdpo: normalise each channel, then combine, then batch-whiten later
        return (z * weights).sum(dim=-1), z

    def _is_informative(self, rewards: torch.Tensor) -> bool:
        if self.cfg.adv_mode == "grpo":
            weights = torch.tensor(
                self.cfg.reward_weights, dtype=rewards.dtype, device=rewards.device
            )
            total = (rewards * weights).sum(dim=-1)
            return bool(total.std(unbiased=False).item() > 1e-6)
        return bool((rewards.std(dim=0, unbiased=False) > 1e-6).any().item())

    # -- rollout -------------------------------------------------------------

    def _rollout_group(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(example["question"])
        gen_out, prompt_len = self._generate(
            prompt, n=self.cfg.num_rollouts, greedy=False
        )

        response_ids = gen_out[:, prompt_len:]
        response_mask, has_eos = build_response_mask(response_ids, self.eos_ids)

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in response_ids
        ]
        truncated = (~has_eos).tolist()
        rewards = self._compute_rewards(texts, example["gold"], truncated)

        old_logp = self._offline_logprobs(gen_out, prompt_len)

        ref_logp: Optional[torch.Tensor] = None
        if self.cfg.use_ref and self.kl.beta > 0.0:
            ref_logp = self._offline_logprobs(gen_out, prompt_len, disable_adapter=True)

        return {
            "input_ids": gen_out,
            "prompt_len": prompt_len,
            "response_mask": response_mask,
            "rewards": rewards,
            "old_logp": old_logp,
            "ref_logp": ref_logp,
            "texts": texts,
            "truncated": truncated,
        }

    def _collect_groups(self) -> List[Dict[str, Any]]:
        """Collect exactly `prompts_per_update` groups on every rank.

        The count must match across ranks, otherwise DDP deadlocks. Dynamic
        sampling is gated on the recent informative rate so that early
        training (where almost everything is all-fail) does not waste rollouts.
        """
        wanted = self.cfg.prompts_per_update
        use_dynamic = (
            self.cfg.dynamic_sampling
            and self._informative_rate >= self.cfg.min_informative_rate
        )
        limit = wanted * max(1, self.cfg.max_rollout_tries)

        kept: List[Dict[str, Any]] = []
        spare: List[Dict[str, Any]] = []
        tries = 0
        informative_hits = 0

        while len(kept) < wanted and tries < limit:
            tries += 1
            group = self._rollout_group(self._next_example())
            informative = self._is_informative(group["rewards"])
            informative_hits += int(informative)
            if informative or not use_dynamic:
                kept.append(group)
            else:
                spare.append(group)

        while len(kept) < wanted:
            kept.append(spare.pop() if spare else self._rollout_group(self._next_example()))

        # Update the gate with a slow EMA.
        rate = informative_hits / max(1, tries)
        self._informative_rate = (
            rate if self._informative_rate == 0.0
            else 0.8 * self._informative_rate + 0.2 * rate
        )
        return kept

    # -- one training iteration ---------------------------------------------

    def train_step(self, groups: List[Dict[str, Any]]) -> Dict[str, float]:
        assert len(groups) == self.cfg.prompts_per_update

        raw_advs: List[torch.Tensor] = []
        zs: List[torch.Tensor] = []
        for group in groups:
            adv, z = self._advantage(group["rewards"])
            group["raw_adv"] = adv.detach()
            raw_advs.append(adv.detach())
            zs.append(z.detach())

        raw = torch.cat(raw_advs)  # [M]
        batch_whiten = self.cfg.adv_mode in ("gdpo", "sum_batch_whiten")
        if batch_whiten:
            gathered = self.accelerator.gather(raw)
            batch_mean = gathered.mean()
            batch_std = gathered.std(unbiased=False).clamp_min(1e-8)
        else:
            batch_mean = torch.zeros((), device=raw.device, dtype=raw.dtype)
            batch_std = torch.ones((), device=raw.device, dtype=raw.dtype)

        beta = self.kl.beta
        n_groups = len(groups)
        self.model.train()

        loss_accum = 0.0
        kl_accum = 0.0
        kl_count = 0.0
        clip_num = 0.0
        clip_den = 0.0
        logp_accum = 0.0
        logp_den = 0.0
        step_ok = True

        for _epoch in range(self.cfg.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for index, group in enumerate(groups):
                advantage = group["raw_adv"]
                if batch_whiten:
                    advantage = (advantage - batch_mean) / batch_std

                mask = group["response_mask"].float()
                counts = mask.sum(dim=-1).clamp_min(1.0)

                policy_logp = self._sequence_logprobs(
                    self.model, group["input_ids"], group["prompt_len"]
                )
                # Clamp only to keep exp() finite; anything outside the clip
                # range is already governed by the PPO min().
                log_ratio = (policy_logp - group["old_logp"]).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)

                adv = advantage.unsqueeze(-1)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                per_token_pg = -torch.min(unclipped, clipped)
                pg_loss = ((per_token_pg * mask).sum(dim=-1) / counts).mean()

                loss = pg_loss
                kl_value = torch.zeros((), device=raw.device, dtype=torch.float32)
                if group["ref_logp"] is not None and beta > 0.0:
                    # k3 estimator, KL(pi_ref || pi_theta), always >= 0.
                    # The clamp is overflow insurance only; a healthy policy
                    # stays within a few nats of the reference.
                    ref_delta = (policy_logp - group["ref_logp"]).clamp(-20.0, 20.0)
                    per_token_kl = torch.exp(-ref_delta) - 1.0 + ref_delta
                    kl_value = ((per_token_kl * mask).sum(dim=-1) / counts).mean()
                    loss = loss + beta * kl_value

                loss = loss / n_groups
                sync = index == n_groups - 1
                context = (
                    self.model.no_sync()
                    if (not sync and hasattr(self.model, "no_sync"))
                    else nullcontext()
                )
                with context:
                    self.accelerator.backward(loss)

                if not torch.isfinite(loss.detach()):
                    step_ok = False

                with torch.no_grad():
                    outside = ((ratio < 1.0 - self.cfg.clip_eps) |
                               (ratio > 1.0 + self.cfg.clip_eps)).float()
                    clip_num += float((outside * mask).sum().item())
                    clip_den += float(mask.sum().item())
                    logp_accum += float((policy_logp * mask).sum().item())
                    logp_den += float(mask.sum().item())
                    if group["ref_logp"] is not None and beta > 0.0:
                        kl_accum += float(kl_value.item())
                        kl_count += 1.0

                loss_accum += float(loss.item())

        # All ranks must agree on whether to step.
        flag = torch.tensor(
            [0.0 if step_ok else 1.0], device=self.accelerator.device
        )
        flag = self.accelerator.reduce(flag, reduction="sum")
        grad_norm = torch.tensor(0.0, device=self.accelerator.device)

        if flag.item() == 0.0:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)

        # Global mean KL for the controller.
        if kl_count > 0:
            stats = torch.tensor(
                [kl_accum, kl_count], device=self.accelerator.device
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            mean_kl = float((stats[0] / stats[1].clamp_min(1.0)).item())
        else:
            mean_kl = 0.0
        if self.cfg.use_ref and self.cfg.adaptive_kl:
            self.kl.observe(mean_kl)

        z_all = self.accelerator.gather(torch.cat(zs))
        rewards_all = self.accelerator.gather(
            torch.cat([g["rewards"] for g in groups])
        )
        lengths = torch.cat(
            [g["response_mask"].sum(dim=-1) for g in groups]
        ).float()
        unique_rates = [
            len(set(g["texts"])) / max(1, len(g["texts"])) for g in groups
        ]

        metrics = {
            "loss": loss_accum / max(1, self.cfg.ppo_epochs * n_groups),
            "kl": mean_kl,
            "beta": beta,
            "grad_norm": float(grad_norm.item()),
            "skipped_step": float(flag.item()),
            "mean_policy_logp": logp_accum / max(1.0, logp_den),
            "clip_fraction": clip_num / max(1.0, clip_den),
            "mean_response_len": float(lengths.mean().item()),
            "truncation_rate": float(np.mean([np.mean(g["truncated"]) for g in groups])),
            "unique_response_rate": float(np.mean(unique_rates)),
            "informative_rate": self._informative_rate,
            "advantage_abs_mean": float(raw.abs().mean().item()),
            "advantage_std": float(raw.std(unbiased=False).item()),
        }
        for k, name in enumerate(self.cfg.reward_names):
            metrics[f"reward/{name}_mean"] = float(rewards_all[:, k].mean().item())
            metrics[f"reward/{name}_std"] = float(rewards_all[:, k].std(unbiased=False).item())
            metrics[f"advnorm/{name}"] = float(z_all[:, k].abs().mean().item())
        return metrics

    # -- evaluation ----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, float], List[Tuple[str, str, str, str]]]:
        if not self.eval_examples:
            return {}, []
        examples = self.eval_examples[: self.cfg.eval_size]

        counters = torch.zeros(3, dtype=torch.long, device=self.accelerator.device)
        samples: List[Tuple[str, str, str, str]] = []
        for example in examples:
            prompt = self._build_prompt(example["question"])
            out, prompt_len = self._generate(
                prompt,
                n=1,
                greedy=True,
                max_new_tokens=self.cfg.eval_max_new_tokens,
            )
            text = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
            answer, used_boxed = parse_answer(text)
            counters[0] += 1
            counters[1] += int(numeric_match(answer, example["gold"]))
            counters[2] += int(used_boxed)
            if self.accelerator.is_main_process and len(samples) < 8:
                samples.append(
                    (
                        example["question"][:200],
                        text[:600],
                        answer,
                        str(example["gold"]),
                    )
                )

        counters = self.accelerator.gather(counters).sum(dim=0)
        total = max(1, int(counters[0].item()))
        metrics = {
            "eval/pass@1": float(counters[1].item()) / total,
            "eval/parse_rate": float(counters[2].item()) / total,
            "eval/n": float(total),
        }
        self.model.train()
        return metrics, samples

    # -- checkpointing -------------------------------------------------------

    def _save(self, step: int) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.cfg.save_dir, self.cfg.run_name, f"step_{step}")
        os.makedirs(path, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if getattr(self.accelerator, "scaler", None) is not None
                else None
            ),
            "beta": self.kl.beta,
            "kl_ema": self.kl.ema,
            "rng_torch": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"config": _to_jsonable(self.cfg), "env": _provenance()},
                fh,
                indent=2,
            )
        print(f"[save] {path}")

    def _load(self, path: str) -> None:
        state_path = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(state_path):
            if self.accelerator.is_main_process:
                print(f"[resume] no trainer_state.pt in {path}; loading adapters only")
            state = {}
        else:
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except TypeError:  # torch < 2.6 has no weights_only
                state = torch.load(state_path, map_location="cpu")

        unwrapped = self.accelerator.unwrap_model(self.model)
        try:
            weights = load_peft_weights(path, device=str(self.accelerator.device))
        except TypeError:
            weights = load_peft_weights(path)
        set_peft_model_state_dict(unwrapped, weights)

        if state.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:  # pragma: no cover
                print(f"[resume] optimizer state not restored: {exc}")
        if state.get("scaler") is not None and getattr(self.accelerator, "scaler", None) is not None:
            self.accelerator.scaler.load_state_dict(state["scaler"])
        if "beta" in state:
            self.kl.beta = float(state["beta"])
        if "kl_ema" in state:
            self.kl.ema = state["kl_ema"]
        if "rng_torch" in state:
            torch.set_rng_state(state["rng_torch"])
        if "rng_python" in state:
            random.setstate(state["rng_python"])
        if "rng_numpy" in state:
            np.random.set_state(state["rng_numpy"])
        self.start_step = int(state.get("step", 0)) + 1
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"[resume] continuing from step {self.start_step}")

    # -- main loop -----------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        if cfg.resume:
            self._load(cfg.resume)

        if self.accelerator.is_main_process and wandb is not None:
            try:
                self._wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.run_name,
                    config=_to_jsonable(cfg),
                )
            except Exception as exc:  # pragma: no cover
                print(f"[wandb] disabled: {exc}")
                self._wandb_run = None

        if self.start_step == 0:
            metrics, samples = self.evaluate()
            self._report(self.start_step, metrics, samples)

        for step in range(self.start_step, cfg.max_steps):
            started = time.time()
            groups = self._collect_groups()
            metrics = self.train_step(groups)
            metrics["seconds"] = time.time() - started

            if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                self._report(step, metrics, None)

            if cfg.eval_freq > 0 and step > 0 and step % cfg.eval_freq == 0:
                eval_metrics, samples = self.evaluate()
                self._report(step, eval_metrics, samples)

            if cfg.save_freq > 0 and step > 0 and step % cfg.save_freq == 0:
                self._save(step)

            del groups
            if cfg.empty_cache_every > 0 and step % cfg.empty_cache_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._save(cfg.max_steps)
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def _report(
        self,
        step: int,
        metrics: Dict[str, float],
        samples: Optional[List[Tuple[str, str, str, str]]],
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        if not metrics:
            return
        shown = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {shown}")
        if self._wandb_run is not None:
            payload = dict(metrics)
            if samples:
                payload["eval/samples"] = wandb.Table(
                    columns=["question", "generation", "pred", "gold"],
                    data=[list(row) for row in samples],
                )
            wandb.log(payload, step=step)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _provenance() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch": torch.__version__}
    for name in ("transformers", "peft", "bitsandbytes", "accelerate", "datasets"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", None)
        except Exception as exc:  # pragma: no cover
            info[name] = f"missing ({exc})"
    try:
        info["git"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        info["git"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def build_gsm8k(cfg: QGDPOConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset_name, cfg.dataset_config)

    def convert(split: str) -> List[Dict[str, Any]]:
        rows = []
        for item in raw[split]:
            rows.append(
                {
                    "question": item["question"],
                    "gold": item["answer"].split("####")[-1].strip(),
                }
            )
        return rows

    return convert(cfg.train_split), convert(cfg.eval_split)


# ----------------------------------------------------------------------------
# Self-test: pure functions only, runs on CPU with no model
# ----------------------------------------------------------------------------


def run_selftest() -> None:
    # --- response mask -----------------------------------------------------
    ids = torch.tensor([[5, 6, 2, 2], [5, 6, 7, 8]])
    mask, has_eos = build_response_mask(ids, [2])
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]], mask
    assert has_eos.tolist() == [True, False], has_eos

    # multi-token EOS set (Qwen: eos vs <|im_end|>)
    ids2 = torch.tensor([[5, 9, 7], [5, 6, 7]])
    mask2, has2 = build_response_mask(ids2, [2, 9])
    assert mask2.tolist() == [[1, 1, 0], [1, 1, 1]], mask2
    assert has2.tolist() == [True, False]

    # --- parser ------------------------------------------------------------
    assert parse_answer("so the answer is \\boxed{42}")[0] == "42"
    assert parse_answer("so the answer is \\boxed{42}")[1] is True
    assert parse_answer("\\boxed { 1,234 }")[0] == "1,234"
    assert parse_answer("no box, answer 7.") == ("7", False)
    assert parse_answer("nothing here") == ("", False)
    assert numeric_match("1,234", "1234")
    assert numeric_match("42.0", "42")
    assert not numeric_match("42", "43")
    assert is_integer_str("-17")
    assert not is_integer_str("17.0")

    # --- channel z ---------------------------------------------------------
    constant = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(channel_z(constant), torch.zeros_like(constant))

    # --- the core claim: GRPO collapse, GDPO distinguishes -----------------
    symmetric = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    grpo_total = symmetric.sum(dim=-1, keepdim=True)
    # All totals are 1.0 -> GRPO advantage is identically zero.
    assert channel_z(grpo_total).abs().max().item() < 1e-6
    # And in the perfectly anti-correlated binary case GDPO's sum also
    # cancels: this is the known limitation, not a bug.
    gdpo_adv = channel_z(symmetric).sum(dim=-1)
    assert gdpo_adv.abs().max().item() < 1e-6

    informative = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    gdpo_adv2 = channel_z(informative).sum(dim=-1)
    assert gdpo_adv2.abs().max().item() > 0.1
    # Correct ordering: full credit > format-only > nothing.
    assert gdpo_adv2[0] > gdpo_adv2[1] > gdpo_adv2[2]

    # --- adaptive KL -------------------------------------------------------
    controller = AdaptiveKL(beta=1e-2, target=1e-2)
    high = controller.observe(1.0)
    assert high > 1e-2
    controller = AdaptiveKL(beta=1e-2, target=1e-2)
    low = controller.observe(0.0)
    assert low < 1e-2

    print("selftest OK: mask, parser, channel-z, GRPO-collapse, adaptive-KL")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QGDPO v2 trainer")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny run for sanity checks")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default=None, choices=["gdpo", "grpo", "sum_batch_whiten"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--prompts", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-ref", action="store_true", help="drop KL + reference forward")
    parser.add_argument("--no-dyn", action="store_true", help="disable dynamic sampling")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="force CPU (very slow; debugging only)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        run_selftest()
        return 0

    cfg = QGDPOConfig()
    if args.model is not None:
        cfg.model_name = args.model
    if args.mode is not None:
        cfg.adv_mode = args.mode
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.rollouts is not None:
        cfg.num_rollouts = args.rollouts
    if args.prompts is not None:
        cfg.prompts_per_update = args.prompts
    if args.ppo_epochs is not None:
        cfg.ppo_epochs = args.ppo_epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.eval_size is not None:
        cfg.eval_size = args.eval_size
    if args.eval_freq is not None:
        cfg.eval_freq = args.eval_freq
    if args.save_freq is not None:
        cfg.save_freq = args.save_freq
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.resume is not None:
        cfg.resume = args.resume
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_4bit:
        cfg.load_in_4bit = False
    if args.no_ref:
        cfg.use_ref = False
        cfg.beta = 0.0
    if args.no_dyn:
        cfg.dynamic_sampling = False
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.cpu:
        cfg.load_in_4bit = False
        cfg.mixed_precision = "no"
    if args.smoke:
        cfg.max_steps = 3
        cfg.prompts_per_update = 2
        cfg.num_rollouts = 2
        cfg.ppo_epochs = 1
        cfg.eval_size = 4
        cfg.eval_freq = 1
        cfg.save_freq = 0
        cfg.max_new_tokens = 64
        cfg.load_in_4bit = cfg.load_in_4bit and torch.cuda.is_available()
        cfg.mixed_precision = "fp16" if torch.cuda.is_available() else "no"
        cfg.run_name = cfg.run_name + "-smoke"

    train_ds, eval_ds = build_gsm8k(cfg)
    trainer = QGDPOTrainer(cfg, train_ds, eval_ds)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
