# medmnist_adveserial_purification
DAE-based adversarial purification of medical images using MedMNIST datasets

# Evaluating Convolutional Denoising Autoencoders for Adversarial Purification of Medical Images Across MedMNIST Datasets

## Overview

This repository contains the implementation of a research project investigating the use of Convolutional Denoising Autoencoders (DAEs) for adversarial purification of medical images.

The framework evaluates image purification and classification performance across three MedMNIST datasets.

## Datasets

- PneumoniaMNIST
- OrganAMNIST
- ChestMNIST

The datasets are obtained from the MedMNIST collection. Please refer to the official [MedMNIST repository](https://github.com/MedMNIST/MedMNIST) for dataset access and licensing information.

## Methodology

The proposed framework includes:

1. Medical image preprocessing and normalization.
2. ResNet-18 classifier training.
3. Adversarial attack simulation.
4. Convolutional Denoising Autoencoder training.
5. Adversarial image purification.
6. Classification and performance evaluation.

## Adversarial Attacks

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Momentum Iterative FGSM (MI-FGSM)
- One-Pixel Attack

## Evaluation Metrics

- Classification accuracy / robust accuracy
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- MedMNIST

## Repository Structure

The source code is organized into preprocessing, classifier training, adversarial attacks, denoising autoencoder, purification, and evaluation modules.

## Reproducibility

Installation instructions, dataset preparation, model configuration, and execution steps will be provided with the corresponding source code.

## Research Notice
This repository accompanies an academic research project. Please cite the associated research publication if you use this implementation, once its publication details are available.
This repository accompanies an academic research project. Please cite the associated research publication if you use this implementation, once its publication details are available.
