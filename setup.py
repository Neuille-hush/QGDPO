from setuptools import setup, find_packages

setup(
    name="qgdpo",
    version="0.2.0",
    description="Quantized Gradient-Descent Preference Optimization with Native QLoRA",
    author="Neuille-hush",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
    ],
)
