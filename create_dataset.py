import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing images
DATA_DIR = './data'

# Lists to store data and labels
data = []
labels = []

# Loop through each subdirectory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if not a directory
    if not os.path.isdir(dir_path):
        continue

    # Loop through each image in the subdirectory
    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        # Read and process the image
        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all x and y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates and store them
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the processed data and label
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
