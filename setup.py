from setuptools import setup, find_packages

setup(
    name="qgdpo",
    version="0.1.0",
    description="Quantized Gradient-Descent Preference Optimization for SLMs",
    author="Neuille-hush",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
)
