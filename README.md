# Code repository for DDIR

**DDIR: Domain-Disentangled Invariant Representation Learning for Tailored Predictions**

Official PyTorch implementation of DDIR.

This repository contains the PyTorch implementation of DDIR (Domain-Disentangled Invariant Representation Learning). The project is built on top of the SWAD and DomainBed codebases.

This work has been accepted by Knowledge-Based Systems.

## Model Overview
Traditional training methods often struggle to scale effectively with large datasets due to significant distributional differences. Domain generalization (DG) aims to address the challenge of generalizing across multiple-source domains and improving performance in unknown target domains. While many DG methods, such as domain-invariant representation (DIR) learning, excel in handling significant distribution shifts, they often sacrifice performance on in-distribution (IID) data. This trade-off is crucial in real-world applications with uncertain distribution shifts, spanning both out-of-distribution (OOD) and IID scenarios. 

**Core Contributions:**

To address this limitation, we propose Domain-Disentangled Invariant Representation (DDIR) learning, which:

- Analyzing the limitations of DIR methods in IID and OOD scenarios and proposing DDIR learning.
- Designing a single-backbone framework for DDIR learning, which separates and preserves DOI information without redundancy.
- Proposing a DOI selection strategy to adapt to individual target instances without requiring knowledge of the target domain distribution.

An overall pipeline of the proposed Domain-Disentangled Invariant Representation learning framework:

![Method](https://github.com/FByyyyuan/DDIR/blob/main/DDIR.png)




## How to Run

How to Run
The `train.py` script conducts multiple leave-one-out cross-validations for all target domains. To train the model, simply run the following command:

```bash
python train.py
```
## Citation

If you find this code useful for your research, we would appreciate it if you could cite our paper as follows:

BibTeX:
```bash
cite
```
