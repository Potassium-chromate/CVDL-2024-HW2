# CVDL-2024-HW2
## Overview
This repository contains the second homework assignment for the CVDL course. The project is divided into two parts
1. VGG19 Training: Implementation and training of the VGG19 model to classify images from the CIFAR-10 dataset.
2. CuGAN Training: Training of a Conditional GAN (CuGAN) to generate handwritten digits similar to those in the MNIST dataset.

This repository demonstrates skills in deep learning model training and evaluation, as well as generative modeling for dataset augmentation.

## Files Overview
- `Q1.py`:
User interface for the first part of the homework, including image loading, augmentation, model structure visualization, and inference for the VGG19 model.
- `Q1_lib.py`:
Contains the underlying functions for handling the actions triggered by the buttons in Q1.py.
- `Q1_train.py`:
Script for training the VGG19 model on the CIFAR-10 dataset and saving the trained weights.
- `Q2.py`:
User interface for the second part of the homework, focusing on generating images using a Conditional GAN (CuGAN).
- `Q2_lib.py`:
Provides the implementation of functions corresponding to the actions performed in Q2.py.
- `Q2_train.py`:
Script for training the Conditional GAN (CuGAN) on the MNIST dataset and saving the trained weights.

## Usage Instructions
1. Generate VGG19 Weights for CIFAR-10 Classification
    - Run `Q1_train.py` to train the VGG19 model on the CIFAR-10 dataset.
    - This will save the trained model weights as `vgg19_cifar10.pth`.
2. Perform Tasks for the First Part of the Homework
    - Run `Q1.py` to load the user interface for the first part.
    - Use the interface to:
      - Load and augment images.
      - Display the VGG19 model structure.
      - View accuracy and loss curves.
      - Perform inference using the trained VGG19 model.
3. Generate GAN Weights for MNIST Generation
    - Run `Q2_train.py` to train the Conditional GAN (CuGAN) on the MNIST dataset.
    - This will save the trained generator model weights as `generator.pth`.
4. Perform Tasks for the Second Part of the Homework
    - Run `Q2.py` to load the user interface for the second part.
    - Use the interface to:
      - Visualize the process of generating MNIST-like images.
      - Interact with the GAN-based model for dataset augmentation.
      
## Dependencies
- Python 3.9.19
- numpy 1.23.5
- PyQt5 5.15.10
- torch 2.5.1
- torchaudio 2.5.1
- torchsummary 1.5.1
- torchvision 0.20.1
- tqdm 4.67.1
