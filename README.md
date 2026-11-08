QGDPO

<p align="center">
  **Quantized Group reward-Decoupled Normalization Policy Optimization**  
  NVIDIA's GDPO adapted for a single 16 GB NVIDIA T4 GPU.
</p><p align="center">
  <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  <img alt="NVIDIA T4 16GB" src="https://img.shields.io/badge/GPU-NVIDIA_T4_16GB-76B900.svg">
  <img alt="4-bit NF4" src="https://img.shields.io/badge/Quantization-4--bit_NF4-orange.svg">
</p>QGDPO is a lightweight implementation of "NVIDIA's GDPO" (https://github.com/NVlabs/GDPO), designed to make multi-reward policy optimization practical on a single NVIDIA T4 GPU with 16 GB of VRAM.

It combines 4-bit NF4 quantization, QLoRA, and GDPO-style reward normalization to reduce memory requirements.

GDPO vs. GRPO

GRPO combines multiple rewards into a single scalar before normalization. This can cause reward signals with different scales or distributions to dominate the optimization process.

GDPO addresses this by:

1. Normalizing each reward channel independently.
2. Combining normalized rewards using configurable weights.
3. Whitening the resulting advantages across the training batch.

"GDPO vs GRPO comparison" (docs/images/gdpo-vs-grpo.png)

Reward design

QGDPO uses four independent reward channels:

- Format: Was the answer enclosed in "\boxed{}"?
- Correctness: Did the numerical answer match the expected answer?
- Integer: Was the answer an integer?
- Completion: Did the response finish without truncation?

Default reward weights: "(1.0, 1.0, 0.5, 0.5)".

"QGDPO reward grid" (docs/images/reward-grid.png)

Memory-efficient setup

Component| Configuration
Base model| Qwen2.5-1.5B-Instruct
Quantization| 4-bit NF4
Fine-tuning| QLoRA
GPU target| NVIDIA T4
VRAM| 16 GB

Installation

git clone https://github.com/Neuille-hush/QGDPO.git
cd QGDPO
pip install -r requirements.txt

Usage

Start training:

accelerate launch train.py

Run a quick sanity check:

accelerate launch train.py --smoke

Run the GRPO baseline:

accelerate launch train.py --mode grpo

Run the self-tests:

python train.py --selftest

Limitations

QGDPO is an experimental implementation. Performance and convergence depend on the model, dataset, reward design, hyperparameters, and available hardware.

Acknowledgements

Based on "NVIDIA's GDPO" (https://github.com/NVlabs/GDPO).

Liu et al., GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization, 2026.

"Paper: arXiv:2601.05242" (https://arxiv.org/abs/2601.05242).

License

Licensed under the "Apache License 2.0" (LICENSE).
