import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json

DATA_PATH = "landmark_data"

# ✅ SORT classes (VERY IMPORTANT)
classes = sorted(os.listdir(DATA_PATH))
print("Classes:", classes)

X = []
y = []

# Load data
for idx, class_name in enumerate(classes):
    class_path = os.path.join(DATA_PATH, class_name)

    for file in os.listdir(class_path):
        data = np.load(os.path.join(class_path, file))

        # ✅ Ensure correct shape
        if data.shape[0] == 63:
            X.append(data)
            y.append(idx)
        else:
            print(f"⚠️ Skipped wrong shape: {file}")

X = np.array(X)
y = to_categorical(y)

print("Total samples:", len(X))
print("Number of classes:", len(classes))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("landmark_model.h5")

# ✅ Save class names (VERY IMPORTANT)
with open("class_names.json", "w") as f:
    json.dump(classes, f)

print("✅ Training Completed")