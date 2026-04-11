from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.13",
        "transformers>=4.30",
        "pandas>=1.5",
        "nltk>=3.8",
        "rouge-score>=0.1.2",
        "PyYAML>=6.0",
        "tqdm>=4.65",
        "numpy>=1.24",
    ],
)
