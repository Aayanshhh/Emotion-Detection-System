# Emotion Detection System 🎭

This repository contains an Emotion Detection System built using ResNet50, a state-of-the-art deep learning model for image classification tasks. The system can analyze facial expressions in real-time or from images and classify them into various emotional states. The project is deployed using Gradio for an interactive and user-friendly interface

## Features 🚀

- **Real-time Emotion Detection**: Detects emotions from live webcam feed or uploaded images with high accuracy..
- **Data Augmentation**: Utilizes techniques like random rotations, zooming, shifting, and flipping to improve model generalization.
- **Interactive Web Interface**: Built with **Gradio**, allowing users to upload images and view predictions in real-time.

## Model Architecture 🧠

- **Base Model**: **ResNet (Residual Neural Network)**, a powerful CNN architecture known for its ability to train very deep networks by using residual connections to avoid vanishing gradients.
- **Layers**: The ResNet model consists of multiple convolutional layers with identity shortcut connections (residuals) to skip layers and improve gradient flow.
- **Pretrained Weights**: The model uses pretrained weights from the ImageNet dataset for better initial performance and faster convergence.
- **Fine-tuning**: The top layers of ResNet have been fine-tuned to classify emotions specifically.
- **Activation Functions**: ReLU activation for the hidden layers and softmax activation for the output layer.
- **Optimizer**: Adam optimizer is used for efficient training.
- **Loss Function**: Categorical cross-entropy is used for multi-class classification of emotions.

## Technologies Used 🛠️

- **Deep Learning Framework**: TensorFlow, Keras
- **Machine Learning Model**: ResNet (Residual Neural Networks)
- **Frontend**: Gradio for the user interface
- **Deployment**: Hosted using Gradio for real-time interaction


