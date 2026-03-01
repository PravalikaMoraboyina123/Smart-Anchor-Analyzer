import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ==============================
# 1. PARAMETERS
# ==============================
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15

train_path = "dataset/fer2013/train"
test_path = "dataset/fer2013/test"

# ==============================
# 2. DATA PREPROCESSING
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

# ==============================
# 3. CNN MODEL
# ==============================

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(7, activation='softmax'))

# ==============================
# 4. COMPILE MODEL
# ==============================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 5. TRAIN MODEL
# ==============================

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# ==============================
# 6. SAVE MODEL
# ==============================

model.save("emotion_model.h5")

print("✅ Model training completed successfully!")
print("Model saved as emotion_model.h5")