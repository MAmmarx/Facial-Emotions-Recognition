# Facial-Emotions-Recognition

📌 Overview
EmotionNet is a Convolutional Neural Network (CNN) trained on the FER2013 dataset to recognize facial expressions in grayscale images. It can classify images into 7 different emotion categories such as Happy, Sad, Angry, and Surprised. Built using Keras, this project demonstrates core deep learning techniques for image classification, including data preprocessing, model building, training, and evaluation.

🧑‍💻 Objective
The goal of this project is to develop a CNN that can accurately recognize human emotions from facial images. The model is trained on real-world data and is capable of classifying 48x48 pixel grayscale images into multiple emotion classes using supervised learning.

🛠️ Features

✅ 1. Data Processing

Loads the FER2013 dataset from a CSV file.

Parses pixel strings into image arrays and reshapes them to (48, 48, 1).

Normalizes pixel values to the range [0, 1].

One-hot encodes emotion labels for classification.

✅ 2. Model Architecture

Built with a Sequential CNN using:

Two Conv2D layers with ReLU activation.

MaxPooling2D to reduce spatial dimensions.

Dropout layers to reduce overfitting.

Dense layers with a final Softmax output layer for 7 classes.

Compiled using the Adam optimizer and categorical cross-entropy loss.

Trained over 30 epochs with real-time validation.

✅ 3. Evaluation & Visualization

Evaluates accuracy on a held-out test set.

Plots training vs validation accuracy to monitor learning progress.

Achieves reliable performance on unseen facial expressions.

⚙️ Technologies Used

Python 3

TensorFlow / Keras

NumPy

Pandas

Matplotlib & Seaborn

🧪 Installation & Usage

🔹 Requirements

Python (3.8 or above)

Jupyter Notebook / Google Colab

TensorFlow (pip install tensorflow)

Pandas, NumPy, Matplotlib

🔹 Run the Notebook

bash
Copy
Edit
jupyter notebook Facial_Emotion_Recognition.ipynb
Or open the file directly in Google Colab.

🧑‍🎓 Developed For
Summer Break – Self Learning Project

✍️ About Me
This project was built independently during the summer as a self-learning experience in deep learning. Through it, I gained hands-on knowledge of CNNs, data preprocessing, model training, and interpreting model performance in a real-world image recognition task.
