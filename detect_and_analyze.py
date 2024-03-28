# detect_and_analyze.py
import os
import cv2
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from mtcnn import MTCNN

# Load the sentiment analysis model
with open('model.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

# Create an MTCNN detector
mtcnn_detector = MTCNN()

def preprocess_for_sentiment(face):
    # Your preprocessing code for sentiment analysis
    # This might include resizing, normalization, etc.
    img = Image.fromarray(face).convert('L')  # Convert to grayscale
    img = img.resize((48, 48))
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0
    pass

def analyze_sentiment(frame):
    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    # Perform sentiment analysis on each detected face
    for face in faces:
        # Extract the face from the frame
        x, y, width, height = face['box']
        face_image = frame[y:y+height, x:x+width]

        # Preprocess the face for sentiment analysis
        processed_face = preprocess_for_sentiment(face_image)

        # Perform sentiment analysis using the loaded model
        pred = sentiment_model.predict(processed_face)
        # Assuming 'label' is a list of class labels
        pred_label = label[pred.argmax()] if 'label' in locals() else f"Class {pred.argmax()}"

        # Print or store the result
        print(f"Face Sentiment: {pred_label}")

# Process frames from a folder
frames_folder = 'C:\\Users\\Lenovo\\Desktop\\BE project\\backend'
for frame_filename in os.listdir(frames_folder):
    frame_path = os.path.join(frames_folder, frame_filename)

    # Read the frame
    frame = cv2.imread(frame_path)

    # Analyze sentiment for the current frame
    analyze_sentiment(frame)

