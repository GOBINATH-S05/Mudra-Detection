import os
os.environ["MPLCONFIGDIR"] = "temp_matplotlib"

from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from datetime import datetime   # ✅ NEW

app = Flask(__name__)

# 🔹 Load model
model = load_model("landmark_model/landmark_model.h5", compile=False)

# 🔹 Load class names
with open("landmark_model/class_names.json") as f:
    class_names = json.load(f)

# 🔹 MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# 🔹 Camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 🔹 Log file
LOG_FILE = "mudra_log.csv"

# 🔹 Prevent duplicate logging
last_logged_label = None


# 🔥 Correction function
def get_correction(label, landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)

    for tip in tips[1:]:
        fingers.append(1 if landmarks[tip][1] < landmarks[tip - 2][1] else 0)

    if label == "Anjali":
        if sum(fingers) > 0:
            return "Close all fingers and join palms"

    elif label == "Mushti":
        if sum(fingers) != 0:
            return "Make a tight fist"

    elif label == "Sikharam":
        if fingers[1] == 0:
            return "Raise index finger"
        if sum(fingers[2:]) > 0:
            return "Fold other fingers"

    elif label == "Pathaka":
        if sum(fingers) < 4:
            return "Extend all fingers straight"

    elif label == "Trishulam":
        if fingers[1] == 0 or fingers[2] == 0:
            return "Raise index and middle fingers"

    elif label == "Alapadmam":
        if sum(fingers) < 4:
            return "Spread fingers like a flower"

    elif label == "Mayura":
        return "Adjust ring finger and thumb connection"

    elif label == "Ardhachandran":
        return "Extend thumb and keep other fingers straight"

    elif label == "Nagabandha":
        return "Cross fingers properly"

    return "Good"


# 🔥 Frame generator
def generate_frames():
    global last_logged_label   # ✅ IMPORTANT

    while True:
        ret, frame = camera.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera blocked or not found!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                if len(landmarks) != 21:
                    continue

                data = np.array(landmarks).flatten().reshape(1, -1)

                prediction = model.predict(data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                label = class_names[class_id] if class_id < len(class_names) else "Unknown"
                correction = get_correction(label, landmarks)

                # ✅ TIMESTAMP LOGGING (ONLY WHEN LABEL CHANGES)
                if label != last_logged_label and confidence > 0.7:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(LOG_FILE, "a") as f:
                        f.write(f"{timestamp},{label}\n")

                    last_logged_label = label

                # Draw hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Prediction text
                text = f"{label} ({confidence:.2f})" if confidence > 0.7 else "Unknown"
                cv2.putText(frame, text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                # Correction text
                cv2.putText(frame, correction, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

        else:
            cv2.putText(frame, "No Hand Detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        # Convert frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 🔹 Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 🔥 RUN
if __name__ == "__main__":
    app.run(debug=True)