QGDPO

Quantized Groupwise Direct Preference Optimization

QGDPO is a resource-efficient implementation of Groupwise Direct Preference Optimization (GDPO), combining 4-bit NF4 quantization and QLoRA to explore reinforcement-learning-based language model training on constrained hardware.

The implementation targets a single NVIDIA T4 GPU with 16 GB of VRAM.

---

Overview

Reinforcement learning for language models can be computationally expensive. QGDPO explores reducing this hardware burden through:

- GDPO: Groupwise optimization for reinforcement learning.
- 4-bit NF4 quantization: Reduces base-model memory requirements.
- QLoRA: Enables parameter-efficient fine-tuning.
- T4 optimization: Targets a single GPU with 16 GB of VRAM.

GDPO vs. GRPO

GDPO builds on group-relative optimization while addressing the contribution of multiple reward objectives. Combining reward components without appropriate normalization can allow differences in scale or distribution to distort the optimization signal.

The following diagram illustrates the conceptual distinction between GRPO and GDPO.

"GDPO vs. GRPO comparison" (docs/images/gdpo-vs-grpo.pdf)

Reward Structure

The following figure illustrates the reward structure used to explain the contribution of individual reward components to groupwise optimization.

"GDPO reward grid" (docs/images/reward-grid.pdf)

Quantization and Fine-Tuning

QGDPO combines quantization and parameter-efficient fine-tuning to reduce GPU memory requirements.

Component| Configuration
Base model precision| 4-bit
Quantization| NF4
Fine-tuning| QLoRA
Target GPU| NVIDIA T4
Target VRAM| 16 GB

NF4 Quantization

NormalFloat 4-bit (NF4) provides a compact representation of model weights and is commonly used in QLoRA-based fine-tuning.

QLoRA

QLoRA keeps the base model quantized while training low-rank adapter parameters, reducing memory consumption compared with full-parameter fine-tuning.

Installation

Clone the repository:

git clone https://github.com/Neuille-hush/GDPO.git
cd GDPO

Install dependencies:

pip install -r requirements.txt

Usage

The main training implementation is "train.py".

python train.py

Adjust the model, dataset, optimization, and hardware configuration according to the parameters defined in the training script.

For a 16 GB T4 GPU, monitor memory usage and adjust batch size, sequence length, and generation settings as needed.

Project Structure

QGDPO/
├── train.py
├── README.md
└── docs/
    └── images/
        ├── gdpo-vs-grpo.png
        └── reward-grid.png

Hardware

The target environment is a single NVIDIA T4 GPU with 16 GB of VRAM.

Actual memory consumption depends on model size, sequence length, batch size, generation settings, and training configuration.

Project Goals

- Explore GDPO under constrained GPU memory.
- Combine groupwise optimization with 4-bit NF4 quantization.
- Use QLoRA for parameter-efficient adaptation.
- Investigate the practical trade-offs of quantized reinforcement learning.

Limitations

Performance and memory usage depend on the model, dataset, hyperparameters, and hardware environment. Results should be evaluated empirically. No benchmark improvements are claimed without experimental measurements.

Acknowledgements

Inspired by NVIDIA's Groupwise Direct Preference Optimization (GDPO).

References

- "NVIDIA GDPO" (https://github.com/NVlabs/GDPO)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (https://arxiv.org/abs/2305.14314)
- "LoRA: Low-Rank Adaptation of Large Language Models" (https://arxiv.org/abs/2106.09685)

License

See the repository's "LICENSE" file for licensing information.
