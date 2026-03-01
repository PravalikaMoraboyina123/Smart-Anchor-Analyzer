import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from collections import Counter
from moviepy.editor import VideoFileClip
import whisper
from textblob import TextBlob
import imageio_ffmpeg

# Fix FFmpeg path for Whisper
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if not exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load models
emotion_model = load_model("emotion_model.h5")
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
whisper_model = whisper.load_model("tiny")

# Store results
history_data = []
latest_result = {}

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("home.html")

# ---------------- ANALYZE PAGE ----------------
@app.route('/analyze')
def analyze_page():
    return render_template("analyze.html")

# ---------------- PROCESS VIDEO ----------------
@app.route('/process', methods=['POST'])
def process():

    file = request.files['video']

    if file.filename == "":
        return redirect(url_for('analyze_page'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # ---------- FACE ANALYSIS ----------
    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(fps, 1)

    emotion_counts = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48,48))
                face = face / 255.0
                face = np.reshape(face, (1,48,48,1))

                pred = emotion_model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(pred)]
                emotion_counts.append(emotion)

        frame_count += 1

    cap.release()

    emotion_counter = Counter(emotion_counts)
    total = sum(emotion_counter.values()) or 1

    confidence_face = (
        emotion_counter.get("neutral",0) +
        emotion_counter.get("happy",0)
    ) / total * 100

    stress_face = (
        emotion_counter.get("angry",0) +
        emotion_counter.get("fear",0)
    ) / total * 100

    face_score = confidence_face - stress_face

    # ---------- AUDIO ANALYSIS ----------
    clip = VideoFileClip(filepath)

    if clip.audio is not None:
        clip.audio.write_audiofile("audio.wav", verbose=False, logger=None)
        result = whisper_model.transcribe("audio.wav")
        text = result["text"]

        sentiment = TextBlob(text).sentiment.polarity

        fillers = ["um","uh","like","actually","basically","you know"]
        filler_count = sum(text.lower().count(w) for w in fillers)
        voice_conf = max(0, 100 - filler_count * 5)
    else:
        text = "No audio detected."
        sentiment = 0
        voice_conf = 0

    # ---------- FINAL SCORE ----------
    final_score = (face_score*0.5) + (voice_conf*0.3) + (sentiment*20)

    global latest_result
    latest_result = {
        "face_score": round(face_score,2),
        "voice_conf": voice_conf,
        "sentiment": round(sentiment,2),
        "final_score": round(final_score,2),
        "transcript": text,
        "emotions": emotion_counter
    }

    history_data.append({
        "video": file.filename,
        "score": round(final_score,2)
    })

    return render_template("analyze.html", result=latest_result)

# ---------------- ANALYTICS PAGE ----------------
@app.route('/analytics')
def analytics():
    if not latest_result:
        return redirect(url_for('analyze_page'))
    return render_template("analytics.html", data=latest_result)

# ---------------- HISTORY PAGE ----------------
@app.route('/history')
def history():
    return render_template("history.html", history=history_data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)