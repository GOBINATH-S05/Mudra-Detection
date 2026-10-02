import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json

# Load trained model
model = load_model("landmark_model/landmark_model.h5")

# Load class names
with open("landmark_model/class_names.json") as f:
    class_names = json.load(f)

print("Loaded classes:", class_names)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)


# 🔥 Correction function
def get_correction(label, landmarks):
    tips = [4, 8, 12, 16, 20]

    fingers = []

    # Thumb
    fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)

    # Other fingers
    for tip in tips[1:]:
        fingers.append(1 if landmarks[tip][1] < landmarks[tip - 2][1] else 0)

    # Corrections
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Extract landmarks (RAW - no normalization)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            if len(landmarks) != 21:
                continue

            data = np.array(landmarks).flatten().reshape(1, -1)

            # Prediction
            prediction = model.predict(data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            # Safe label
            if class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = "Unknown"

            # Get correction
            correction = get_correction(label, landmarks)

            # Display main text
            if confidence > 0.7:
                text = f"{label} ({confidence:.2f})"
            else:
                text = "Unknown"

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show prediction
            cv2.putText(frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            # Show correction
            cv2.putText(frame, correction, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

    else:
        cv2.putText(frame, "No Hand Detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    cv2.imshow("Mudra Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()