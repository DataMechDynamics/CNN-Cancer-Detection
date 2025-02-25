CNN-Based Cancer Detection

Project Overview

This project focuses on developing a Convolutional Neural Network (CNN) to detect cancerous cells from medical images. The primary goal is to leverage deep learning techniques to classify images as either cancerous or non-cancerous with high accuracy. The dataset used in this project comes from Kaggle's Mini-Project on Cancer Detection, which contains labeled images for training and testing the model.

Dataset

The dataset is sourced from Kaggle and contains images categorized into different classes based on cancer presence. It has been preprocessed and structured in a way suitable for deep learning model training. The dataset is split into training, validation, and test sets to ensure effective model evaluation.

Implementation Details

The following steps were taken to develop the CNN model:

Data Preprocessing:

Loaded and resized images using OpenCV and PIL.

Augmented the data using TensorFlow's ImageDataGenerator.

Normalized pixel values to improve training efficiency.

Model Architecture:

Implemented a CNN using TensorFlow and Keras.

Used multiple convolutional layers with ReLU activation functions.

Applied max pooling to reduce spatial dimensions.

Added fully connected layers with dropout for regularization.

Model Training:

Split the dataset using train_test_split.

Compiled the model with categorical cross-entropy loss and Adam optimizer.

Trained the model with real-time data augmentation.

Evaluation & Results:

Evaluated the model using accuracy, precision, recall, and F1-score.

Generated a confusion matrix and classification report.

Visualized training performance using Matplotlib and Seaborn.

Installation & Usage

Prerequisites

Ensure you have the following Python libraries installed:

pip install tensorflow numpy pandas matplotlib seaborn opencv-python pillow scikit-learn

Running the Model

To train the model, execute the following command:

python train_model.py

Modify hyperparameters, dataset paths, and augmentation techniques in the script as needed.

Project Structure

|-- cnn-cancer-detection
    |-- data/                  # Directory containing the dataset
    |-- models/                # Saved trained models
    |-- notebooks/             # Jupyter notebooks for exploratory analysis
    |-- src/                   # Source files for training and evaluation
    |-- train_model.py         # Script for training the CNN model
    |-- README.md              # Project documentation

Conclusion

This project demonstrates the potential of deep learning in medical imaging for cancer detection. By leveraging CNNs, we were able to achieve high accuracy in classification tasks. Further improvements can be made by experimenting with different architectures, transfer learning, and hyperparameter tuning.
