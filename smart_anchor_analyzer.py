import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
from moviepy.editor import VideoFileClip
import whisper
from textblob import TextBlob
import imageio_ffmpeg

# ===============================
# FIX FFMPEG (No PATH Needed)
# ===============================
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

# ===============================
# LOAD EMOTION MODEL
# ===============================
print("Loading emotion model...")
emotion_model = load_model("emotion_model.h5")

emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ===============================
# VIDEO INPUT
# ===============================
video_path = input("Enter video file name (example: anchor_video.mp4): ")

if not os.path.exists(video_path):
    print("❌ Video not found.")
    exit()

# ===============================
# PART 1: FAST FACE ANALYSIS
# ===============================
print("\nAnalyzing Facial Emotions...")

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(fps, 1)   # process 1 frame per second

emotion_counts = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

            prediction = emotion_model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            emotion_counts.append(emotion)

    frame_count += 1

cap.release()

if len(emotion_counts) == 0:
    print("❌ No faces detected.")
    exit()

emotion_counter = Counter(emotion_counts)
total_frames = sum(emotion_counter.values())

confidence_face = (emotion_counter.get("neutral", 0) +
                   emotion_counter.get("happy", 0)) / total_frames * 100

stress_face = (emotion_counter.get("angry", 0) +
               emotion_counter.get("fear", 0)) / total_frames * 100

face_score = confidence_face - stress_face

print("Face Confidence Score:", round(face_score, 2))

# ===============================
# PART 2: VOICE ANALYSIS (FAST)
# ===============================
print("\nExtracting Audio...")

clip = VideoFileClip(video_path)
clip.audio.write_audiofile("audio.wav", verbose=False, logger=None)

print("Loading Whisper (tiny model for speed)...")
whisper_model = whisper.load_model("tiny")

print("Transcribing...")
result = whisper_model.transcribe("audio.wav")
text = result["text"]

print("\nTranscript:")
print(text)

# ===============================
# SENTIMENT ANALYSIS
# ===============================
analysis = TextBlob(text)
sentiment_score = analysis.sentiment.polarity

print("\nSentiment Score:", round(sentiment_score, 2))

# ===============================
# FILLER WORD DETECTION
# ===============================
fillers = ["um", "uh", "like", "actually", "basically", "you know"]
lower_text = text.lower()
filler_count = sum(lower_text.count(word) for word in fillers)

voice_confidence = max(0, 100 - filler_count * 5)

print("Filler Words Count:", filler_count)
print("Voice Confidence Score:", voice_confidence)

# ===============================
# FINAL PERFORMANCE SCORE
# ===============================
final_score = (face_score * 0.5) + (voice_confidence * 0.3) + (sentiment_score * 20)

print("\n==============================")
print("SMART ANCHOR FINAL REPORT")
print("==============================")
print("Face Score:", round(face_score, 2))
print("Voice Confidence:", voice_confidence)
print("Sentiment Impact:", round(sentiment_score * 20, 2))
print("\n🔥 FINAL PERFORMANCE SCORE:", round(final_score, 2))
print("==============================")