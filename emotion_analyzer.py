import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import os

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = load_model("emotion_model.h5")

emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']

# Better face detection settings
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# SELECT INPUT MODE
# ==============================
print("\n========== SMART ANCHOR ANALYZER ==========")
print("1 - Webcam")
print("2 - Video File")

choice = input("Enter 1 or 2: ")

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    video_path = input("Enter video file name (example: anchor_video.mp4): ")
    
    if not os.path.exists(video_path):
        print("❌ Video file not found in project folder.")
        exit()
        
    cap = cv2.VideoCapture(video_path)
else:
    print("❌ Invalid choice.")
    exit()

emotion_counts = []
timeline = []
frame_number = 0

print("\nPress 'q' to stop analysis...\n")

# ==============================
# PROCESS VIDEO FRAMES
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improved face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        emotion_counts.append(emotion)
        timeline.append((frame_number, emotion))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Smart Anchor Emotion Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==============================
# PERFORMANCE REPORT
# ==============================
if len(emotion_counts) == 0:
    print("\n⚠ No faces detected in video.")
    exit()

counter = Counter(emotion_counts)
total = sum(counter.values())

print("\n========== EMOTION ANALYSIS REPORT ==========")

for emotion, count in counter.items():
    percentage = (count / total) * 100
    print(f"{emotion}: {count} frames ({percentage:.2f}%)")

# Performance Metrics Logic
confidence_score = (counter.get("neutral",0) + counter.get("happy",0)) / total * 100
stress_score = (counter.get("angry",0) + counter.get("fear",0)) / total * 100
positivity_score = (counter.get("happy",0)) / total * 100

performance_score = (confidence_score + positivity_score) - stress_score

print("\nTotal Frames Analyzed:", total)
print("Confidence Score:", round(confidence_score,2))
print("Stress Score:", round(stress_score,2))
print("Positivity Score:", round(positivity_score,2))
print("\n🔥 Final Performance Score:", round(performance_score,2))

print("\nAnalysis Completed Successfully ✅")