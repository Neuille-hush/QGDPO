#QGDPO

<p align="center">
  <strong>Quantized Group reward-Decoupled Normalization Policy Optimization</strong><br>
  <em>Memory-efficient multi-reward policy optimization for a single 16 GB NVIDIA T4 GPU.</em>
</p>

<p align="center">
  <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  <img alt="GPU: NVIDIA T4 16 GB" src="https://img.shields.io/badge/GPU-NVIDIA_T4_16GB-76B900.svg">
  <img alt="4-bit NF4 quantization" src="https://img.shields.io/badge/Quantization-4--bit_NF4-orange.svg">
  <img alt="QLoRA fine-tuning" src="https://img.shields.io/badge/Fine--tuning-QLoRA-informational.svg">
</p>

---

Overview

QGDPO is an experimental, lightweight implementation of Group reward-Decoupled Normalization Policy Optimization (GDPO) (NVlabs/GDPO), adapted to run on a single NVIDIA T4 GPU with 16 GB of VRAM.

It combines 4-bit NF4 quantization, QLoRA, and multi-channel reward normalization to make reinforcement-learning fine-tuning more practical on constrained hardware.

At a glance

Component Configuration
Base model Qwen2.5-1.5B-Instruct
Optimization GDPO (with a GRPO baseline)
Quantization 4-bit NF4
Fine-tuning QLoRA
Target GPU NVIDIA T4
GPU memory 16 GB

---

How QGDPO works

docs/images/gdpo-vs-grpo.png

The figure above visualizes the main stages of QGDPO: generating multiple responses, evaluating them with independent reward channels, normalizing each reward separately, combining the normalized signals, and updating the policy with memory-efficient QLoRA fine-tuning.

GDPO: why decouple rewards?

In standard GRPO, multiple reward signals are typically combined into one scalar before advantage normalization. When reward channels have different scales or distributions, one signal can dominate the others.

GDPO instead:

1. Normalizes each reward channel independently.
2. Combines the normalized signals using configurable weights.
3. Whitens the resulting advantages across the training batch.

---

Reward design

docs/images/reward-grid.png

QGDPO uses four independent reward channels:

Reward What it checks
Format The answer uses the required \boxed{} format.
Correctness The numerical answer matches the expected result.
Integer The answer is an integer.
Completion The response finishes without being truncated.

The default reward weights are (1.0, 1.0, 0.5, 0.5), respectively.

---

Installation

```bash
git clone https://github.com/Neuille-hush/QGDPO.git
cd QGDPO
pip install -r requirements.txt
```

---

Usage

Start training:

```bash
accelerate launch train.py
```

Run a quick smoke test:

```bash
accelerate launch train.py --smoke
```

Run the GRPO baseline:

```bash
accelerate launch train.py --mode grpo
```

Run the self-tests:

```bash
python train.py --selftest
```

---

Notes and limitations

QGDPO is an experimental implementation. Training behavior and results depend on the model, dataset, reward definitions, hyperparameters, and available hardware. The NVIDIA T4 target describes the intended memory-constrained setup; actual runtime and achievable batch sizes will vary.

---

Acknowledgements

This project is based on NVIDIA's GDPO implementation and the paper:

Liu et al., GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization (2026). arXiv:2601.05242

---

License

This project is licensed under the Apache License 2.0.
