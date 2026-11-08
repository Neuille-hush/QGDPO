<h1 align="center">QGDPO</h1>

<p align="center">
  <b>Quantized Group reward-Decoupled Normalization Policy Optimization</b><br>
  NVIDIA's GDPO, shrunk to fit on a single T4 GPU.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="GPU: NVIDIA T4" src="https://img.shields.io/badge/GPU-NVIDIA_T4_16GB-76B900.svg">
  <img alt="Quantization: NF4" src="https://img.shields.io/badge/Quantization-4--bit_NF4-orange.svg">
</p>

---

## What is QGDPO?

QGDPO is built on [NVIDIA's GDPO](https://github.com/NVlabs/GDPO) (Group reward-Decoupled Normalization Policy Optimization), with one sole mission: **make GDPO fit and train on a single NVIDIA T4 GPU (16 GB VRAM).**

It does this by combining GDPO with 4-bit NF4 quantization and QLoRA, so the base model stays small in memory while only lightweight adapter weights are trained.

## Why GDPO?

Group-based RL methods like GRPO sample several responses per prompt and score each one relative to its group. With multiple rewards, GRPO adds them into a single scalar *before* normalizing. Rewards with larger scales or different distributions can then dominate, and different reward combinations can collapse into near-identical advantages.

GDPO fixes this by **decoupling the normalization**:

1. Normalize each reward separately within its group.
2. Combine the normalized rewards using their weights.
3. Normalize the combined advantage across the batch.

Each objective keeps its own contribution to the training signal.

> GDPO is a GRPO-style policy optimization method. It is not Direct Preference Optimization (DPO) and does not use preference pairs.

![GDPO vs. GRPO](docs/images/img.png)

### Reward structure

![Reward grid](docs/images/img2.png)

## Memory-saving setup

| Component | Configuration |
|---|---|
| Base model precision | 4-bit |
| Quantization format | NF4 |
| Fine-tuning | QLoRA |
| Target GPU | NVIDIA T4 |
| Target VRAM | 16 GB |

- **NF4 (NormalFloat 4-bit):** a quantization format designed for normally distributed neural network weights. It stores model parameters compactly.
- **QLoRA:** keeps the base model quantized and trains only low-rank adapters, which cuts memory use compared with full fine-tuning.

## Requirements

- NVIDIA T4 GPU (16 GB VRAM)
- CUDA-compatible PyTorch environment

Actual memory use depends on model size, sequence length, batch size, generation settings, and training configuration.

## Installation

```bash
git clone https://github.com/Neuille-hush/QGDPO.git
cd QGDPO
pip install -r requirements.txt
```

## Usage

The training implementation lives in `train.py`:

```bash
python train.py
```

Set the model, dataset, optimization parameters, and hardware options in the script. On a 16 GB T4, watch memory usage and lower batch size, sequence length, or generation length if you run out of VRAM.

## Repository structure

```
QGDPO/
├── train.py
├── requirements.txt
├── LICENSE
├── README.md
└── docs/
    └── images/
        ├── img.png
        └── img2.png
```

## Limitations

QGDPO is an experimental implementation, not a claim of better performance than full-precision GDPO. Memory usage, convergence, and results depend on the model, dataset, hyperparameters, and hardware, and need empirical testing.

## Acknowledgements

QGDPO is derived from [NVIDIA's GDPO](https://github.com/NVlabs/GDPO). All credit for the algorithm goes to its authors:

> Liu et al., *GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization*, 2026. [arXiv:2601.05242](https://arxiv.org/abs/2601.05242)

## References

- [NVIDIA GDPO](https://github.com/NVlabs/GDPO)
- [GDPO paper](https://arxiv.org/abs/2601.05242)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## License

Licensed under the [Apache License 2.0](LICENSE).
