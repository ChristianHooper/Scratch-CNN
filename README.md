# Scratch-CNN
A modular CNN for detection and classification build with NumPy & Cython from scratch.

## Setup
Create an environment with Conda:
```bash
conda env create --file standard_environment.yml --name scratch
conda activate scratch
```

## Run
Execute the sample pipeline on the test data:
```bash
python src/cnn_base.py
```

## Optional Cython acceleration
Build the `fast_ops` extension with:
```bash
python -m pip install -e .
```
If the extension is not built, the code falls back to the Python implementation. Right now only the max pooling function is supported.
