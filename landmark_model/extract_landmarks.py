import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# Paths
DATASET_PATH = "dataset_images"
SAVE_PATH = "landmark_data"

# 🔴 Clear old data (important)
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)

os.makedirs(SAVE_PATH, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# 🟢 SORTED class names (VERY IMPORTANT)
classes = sorted(os.listdir(DATASET_PATH))
print("Classes:", classes)

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    save_class_path = os.path.join(SAVE_PATH, class_name)

    os.makedirs(save_class_path, exist_ok=True)

    count = 0
    skipped = 0

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Cannot read: {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # ✅ Ensure correct size (63 values)
                if len(landmarks) == 63:
                    np.save(os.path.join(save_class_path, f"{count}.npy"),
                            np.array(landmarks))
                    count += 1
                else:
                    print(f"⚠️ Wrong landmark size in {img_path}")
                    skipped += 1
        else:
            print(f"⚠️ No hand detected: {img_path}")
            skipped += 1

    print(f"✅ {class_name}: {count} saved | {skipped} skipped")

print("\n🎉 Extraction Completed Successfully!")