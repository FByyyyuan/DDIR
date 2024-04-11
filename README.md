# Code repository for DDIR

**Rethinking Domain Invariant Representations: Domain-Disentangled Invariant Representation Learning**

PyTorch implementation of DDIR. This project is built upon *SWAD* and *DomainBed* codebases.

## Environments

Environment details used for our study.

```
        Python: 3.7.13
        PyTorch: 1.11.0
        Torchvision: 0.12.0
        CUDA: 11.3
        CUDNN: 8200
        NumPy: 1.21.6
        PIL: 9.2.0
```

## How to Run

`train.py` script conducts multiple leave-one-out cross-validations for all target domain.
