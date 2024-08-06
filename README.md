# CIFAR-10_model

This project is an implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Table of Contents
- Introduction
- Dataset
- Model Architecture
- Installation
- Usage
- Results
- Contributing

## Introduction
This repository contains a CNN model built using PyTorch to classify images from the CIFAR-10 dataset. The goal is to achieve high accuracy in classifying images into one of the ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Dataset
The CIFAR-10 dataset is publicly available and can be downloaded from the CIFAR-10 website. The dataset is divided into 50,000 training images and 10,000 test images.

## Model Architecture
The model consists of several convolutional layers, max-pooling layers, and fully connected layers. The architecture cinvolves:

- Convolutional Layer
- ReLU Activation
- Max Pooling
- Batch Normalisation
- Fully Connected Layer
- Softmax Output

## Installation
To run this project, you'll need to have Python and the following libraries installed:

- PyTorch
- torchvision
- numpy
- matplotlib

pip install torch torchvision numpy matplotlib

git clone https://github.com/yourusername/CIFAR-10_Model.git
cd CIFAR-10_Model

from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

## Results
After training the model, you should see an accuracy of around XX% on the test dataset. Here are some sample results:

### Metric	Value
- Train Loss	0.45
- Train Accuracy	84.02%
- Test Loss	0.56
- Test Accuracy	81.98%

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
