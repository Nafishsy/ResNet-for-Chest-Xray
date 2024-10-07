# COVID-19 Radiography Classification

This project classifies radiography images into two categories: **COVID-19** and **Normal** using a deep learning model based on a pre-trained ResNet50 architecture.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [How to Use](#how-to-use)
- [Acknowledgments](#acknowledgments)

## Project Overview
The purpose of this project is to build a binary image classification model to distinguish between chest X-rays that show **COVID-19 infection** and those that are **normal**. The model uses **transfer learning** with a ResNet50 backbone and is fine-tuned to classify radiography images.

## Dataset
The dataset used for this project is the **COVID-19 Radiography Dataset**, which contains labeled X-ray images of COVID-19 and normal cases.

- **Dataset Structure**:
    - COVID
    - Normal

Each folder contains X-ray images in `.png` format.

You can download the dataset from [Kaggle COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

## Model Architecture
The model is built using **transfer learning** with the **ResNet50** architecture. The final layers of the network are customized for binary classification.

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Top Layers**: 
    - GlobalAveragePooling2D
    - Dense Layer (1 unit, Sigmoid Activation for binary classification)

### Optimizer and Loss
- Optimizer: Adam (`learning_rate=0.001`)
- Loss Function: `sparse_categorical_crossentropy` for handling integer labels.
- Metrics: `accuracy`, `precision`, `recall`, `F1-score`.

## Preprocessing
- Images are resized to **256x256 pixels**.
- Pixel values are normalized to a range of `[0, 1]`.
- Labels are extracted from the file path names.

```python
image = tf.image.resize(image, (256, 256))
image = image / 255.0
