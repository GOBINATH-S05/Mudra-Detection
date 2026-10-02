import tensorflow as tf

# Paths
train_path = "Small data traing/train"
val_path = "Small data traing/val"

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=(128,128),
    batch_size=32
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_path,
    image_size=(128,128),
    batch_size=32
)

# ✅ SAVE class names BEFORE mapping
class_names = train_data.class_names
print("Classes:", class_names)

# Normalize data
train_data = train_data.map(lambda x,y: (x/255.0, y))
val_data = val_data.map(lambda x,y: (x/255.0, y))

# Improve performance (optional but good)
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    # ✅ FIXED LINE
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("mudra_model.h5")

print("Training Done ✅")