import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*deprecated.*")

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'Fuck You', 1: 'Hello', 2: 'No'}

# Buffer for storing predictions
predictions_buffer = []
buffer_size = 10  # Adjust buffer size as needed

# Track the last time predictions were written
last_write_time = time.time()
write_interval = 1  # Interval in seconds between writes to the file

# Open the text file for writing
with open('predictions.txt', 'w') as file:
    recognized_text = ""  # Variable to store recognized text

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if data_aux:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                    # Add the predicted character to the buffer
                    if not predictions_buffer or predictions_buffer[-1] != predicted_character:
                        predictions_buffer.append(predicted_character)

                    # Update recognized text
                    recognized_text = ' '.join(predictions_buffer)

                    # Write to file if buffer reaches size or time interval has passed
                    current_time = time.time()
                    if len(predictions_buffer) >= buffer_size or (current_time - last_write_time) >= write_interval:
                        # Write current buffer to file with line length control
                        while len(predictions_buffer) > 0:
                            line = []
                            while len(line) < 22 and len(predictions_buffer) > 0:
                                line.append(predictions_buffer.pop(0))
                            file.write(' '.join(line) + ' ')
                        last_write_time = current_time

        # Display the recognized text on the frame
        cv2.putText(frame, recognized_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Detection', frame)
        cv2.putText(frame, 'Press "Q" to Exit :)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write any remaining predictions in the buffer to the file
    if predictions_buffer:
        while len(predictions_buffer) > 0:
            line = []
            while len(line) < 15 and len(predictions_buffer) > 0:
                line.append(predictions_buffer.pop(0))
            file.write(' '.join(line) + ' ')

cap.release()
cv2.destroyAllWindows()
