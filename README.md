# RealTime-Face-Emotion-Detection
A deep learning-based emotion detection system that classifies facial images into Sad or Happy categories. Supports prediction from static images and real-time webcam input using TensorFlow and OpenCV.

## Table of Contents

- [Project Overview]  
- [Features]  
- [Demo]  
- [Installation]  
- [Usage]  
- [Model Training]
- [Folder Structure]
- [Technologies Used] 


---

## Project Overview

Facial emotion recognition is a vital part of human-computer interaction systems. This project focuses on detecting two basic emotions: *Sad* and *Happy* using a convolutional neural network (CNN). The trained model classifies images with reasonable accuracy and can be extended to real-time webcam input.

---

## Features

- Binary classification: Sad vs Happy  
- Accepts input images from the dataset or any image file  
- Real-time webcam emotion detection (optional extension)  
- Visual display of predictions along with confidence scores  

---

## Demo


https://github.com/user-attachments/assets/37184e8e-46a1-4efa-b913-63dda4c5f9af


## Dataset Link

https://www.kaggle.com/datasets/msambare/fer2013


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gayatribakle/emotion-detection.git
   cd emotion-detection
   ```
Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Make sure you have your trained model file named model.h5 in the project directory.

Usage

Place the images you want to test inside a folder (e.g., test_images).

Run the prediction script:
```
python predict_emotion.py --image_path test_images/sample.jpg

```
To test multiple images:
```
python batch_predict.py --folder_path test_images/
```

(Optional) For real-time webcam emotion detection, run:
```
python webcam_emotion_detection.py
```
Model Training

If you want to retrain the model from scratch:

Prepare your dataset with labeled images for 'Sad' and 'Happy'.

Run the training script:
```
python train_model.py
```

The trained model will be saved as model.h5.
```
Folder Structure
emotion-detection/
│
├── train_model.py             # Script to train the model
├── predict_emotion.py         # Script to predict emotion from a single image
├── batch_predict.py           # Script to predict emotion from a folder of images
├── webcam_emotion_detection.py # (Optional) Real-time webcam emotion detection
├── model.h5                   # Trained model file
├── requirements.txt           # Required Python packages
├── README.md                  # This file
├── test_images/               # Sample images for testing
└── dataset/                   # Dataset folder (if included)
```
### Technologies Used

Python 3.x

TensorFlow / Keras

OpenCV (for webcam real-time detection)

NumPy

Matplotlib
