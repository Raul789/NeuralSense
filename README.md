# Brain Tumor Detection with Convolutional Neural Networks (CNN)

## Overview

This project focuses on developing a Convolutional Neural Network (CNN) for the detection of brain tumors from medical imaging. The model is trained to classify brain MRI images as either 'tumor' or 'no tumor' using a custom neural network architecture.

The project includes image preprocessing, data augmentation, and performance evaluation, including metrics like accuracy, precision, recall, F1 score, and ROC AUC. The trained model is capable of classifying MRI images and outputting predictions, with the ability to move correctly classified images into corresponding folders for further analysis.

## Features

- **Image Preprocessing**: Cropping brain contours from MRI images for better focus on the tumor regions.
- **Model Architecture**: CNN-based neural network to classify images.
- **Data Augmentation**: Includes additional augmented data for model robustness.
- **Performance Metrics**: Calculates accuracy, precision, recall, F1 score, and ROC AUC.
- **Model Checkpoints & TensorBoard**: Tracks the model's training progress and saves the best-performing model.
- **Image Classification & Sorting**: Classifies images and moves them to folders based on predicted labels.
