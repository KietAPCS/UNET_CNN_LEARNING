# UNET_CNN_LEARNING

This repository contains an implementation of the UNet Convolutional Neural Network (CNN) for image segmentation tasks. UNet is widely used in the field of medical image processing but can also be applied to any general image segmentation problem.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

## Introduction

The UNet architecture is a fully convolutional neural network that has proven effective in pixel-level image segmentation tasks. It consists of two parts: a contraction path that captures context and a symmetric expansion path that enables precise localization.

This project is designed for:
- Learning the UNet architecture for image segmentation
- Training the model on your own data
- Testing the model on new images

## Features
- Dataset preprocessing and augmentation
- Training with CUDA support for faster computation
- Model checkpointing to save the best model during training
- Evaluation and performance metrics on the test dataset
- Visualization of results including segmented images

## Requirements

To run this project, you'll need the following dependencies:
- Python 3.x
- PyTorch
- CUDA (optional, for GPU acceleration)
- NumPy
- Matplotlib
- OpenCV

## Setups

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/UNET_CNN_LEARNING.git
   cd UNET_CNN_LEARNING
2. Ensure you have CUDA installed for GPU training. You can check this by running the provided script:
   ```bash
   python check_cuda.py

## Training
To train the UNet model, use the following command:
  ```bash 
  python train.py
  ```

## File Structure

The project structure is organized as follows:


Each component is described as follows:
- **data/**: This folder is used to store the input training and testing data.
- **results/**: The folder where training logs, model checkpoints, and results will be saved.
- **saved_images/**: This folder contains the segmented images generated after testing.
- **dataset.py**: Script to handle loading and transformations of the dataset.
- **model.py**: Contains the implementation of the UNet architecture.
- **train.py**: Script to train the UNet model.
- **test.py**: Script to test the trained model on new images.
- **utils.py**: Contains helper functions used across the project.
- **split_data.py**: Script to split the dataset into training and testing sets.
- **check_cuda.py**: Script to check for the availability of CUDA for GPU acceleration.
- **my_checkpoint.pth.tar**: The model checkpoint file saved after training.
- **.gitattributes**: Configuration file for Git Large File Storage (LFS).

## Acknowledgments

- [Original UNet Paper](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox, which introduced the UNet architecture for biomedical image segmentation.
- PyTorch framework for enabling deep learning model development with ease.
- Open-source libraries and tools that made this project possible.
- Special thanks to the contributors and researchers whose work inspired and supported the development of this project.
- Special thanks to Aladdin Persson who guide me in this project
  
