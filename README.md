# Spatiotemporal MAE for Cardiac MRI

Official codebase for evaluating few-shot classification and data leakage in 4D Cardiac MRI using a Spatiotemporal Masked Autoencoder (MAE).

## Overview
Training deep supervised models on small medical cohorts (e.g., N=11) usually leads to severe overfitting. This repository provides an empirical baseline to study this. We use a self-supervised 3D MAE to reconstruct cardiac geometry with a 75% temporal mask, followed by a frozen linear probe to evaluate actual few-shot pathology detection.

## Repository Structure
* `dataset.py`: DICOM loading and strict patient-level splitting.
* `model.py`: 3D Convolutional Encoder/Decoder architecture.
* `train.py`: Self-supervised pre-training loop using masked MSE loss.
* `evaluate.py`: Matplotlib visualization of the reconstructed sequences.
* `linear_probe.py`: SVM downstream evaluation on frozen features.
* `supervised_baseline.py`: A standard 3D CNN provided to demonstrate the overfitting trap on small cohorts.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt