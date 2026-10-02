import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("../landmark_model/landmark_model.h5")

# -------------------------
# Load Data
# -------------------------
data_path = "../landmark_model/landmark_data"

X = []
y = []

class_names = sorted(os.listdir(data_path))

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(data_path, class_name)

    for file in os.listdir(class_folder):
        filepath = os.path.join(class_folder, file)

        data = np.load(filepath)
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

# -------------------------
# Predict
# -------------------------
predictions = model.predict(X)
y_pred = np.argmax(predictions, axis=1)

# -------------------------
# Accuracy
# -------------------------
accuracy = accuracy_score(y, y_pred)
print("\nOverall Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------
# Confusion Matrix (FINAL CLEAN VERSION)
# -------------------------
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,5))

# Black & white for paper
plt.imshow(cm, cmap='gray')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

# Show only index numbers (no class names)
ticks = np.arange(len(class_names))
plt.xticks(ticks)
plt.yticks(ticks)

# Optional: add values inside cells (professional touch)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="black", fontsize=8)

plt.tight_layout()

# Save high-quality image for paper
plt.savefig("confusion_matrix_final.png", dpi=300)

# DO NOT block execution
plt.close()

print("\nConfusion matrix saved as 'confusion_matrix_final.png'")

# -------------------------
# Classification Report (WITHOUT CLASS NAMES)
# -------------------------
print("\nClassification Report (Overall):\n")

report = classification_report(y, y_pred, output_dict=True)

print("Precision:", round(report["weighted avg"]["precision"], 2))
print("Recall:", round(report["weighted avg"]["recall"], 2))
print("F1-score:", round(report["weighted avg"]["f1-score"], 2))