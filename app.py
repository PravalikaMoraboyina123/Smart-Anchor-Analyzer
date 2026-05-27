from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
from moviepy.editor import VideoFileClip
import whisper
from textblob import TextBlob
import imageio_ffmpeg
from werkzeug.utils import secure_filename

# ---------------- APP SETUP ----------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fix FFmpeg path
os.environ["PATH"] += os.pathsep + os.path.dirname(
    imageio_ffmpeg.get_ffmpeg_exe()
)

# ---------------- LOAD MODELS ----------------

emotion_model = load_model("emotion_model.h5")

emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Whisper tiny model
whisper_model = whisper.load_model("tiny")

# ---------------- GLOBAL STORAGE ----------------

history_data = []
latest_result = {}

# ---------------- HOME PAGE ----------------

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

    if 'video' not in request.files:
        return redirect(url_for('analyze_page'))

    file = request.files['video']

    if file.filename == '':
        return redirect(url_for('analyze_page'))

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)

    # ---------------- FACE ANALYSIS ----------------

    cap = cv2.VideoCapture(filepath)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 1

    frame_interval = fps

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
                minNeighbors=4
            )

            for (x, y, w, h) in faces:

                face = gray[y:y+h, x:x+w]

                try:
                    face = cv2.resize(face, (48, 48))

                    face = face / 255.0

                    face = np.reshape(face, (1, 48, 48, 1))

                    prediction = emotion_model.predict(
                        face,
                        verbose=0
                    )

                    emotion = emotion_labels[np.argmax(prediction)]

                    emotion_counts.append(emotion)

                except:
                    continue

        frame_count += 1

    cap.release()

    emotion_counter = Counter(emotion_counts)

    total = sum(emotion_counter.values())

    if total == 0:
        total = 1

    confidence_face = (
        emotion_counter.get("neutral", 0) +
        emotion_counter.get("happy", 0)
    ) / total * 100

    stress_face = (
        emotion_counter.get("angry", 0) +
        emotion_counter.get("fear", 0)
    ) / total * 100

    face_score = confidence_face - stress_face

    # ---------------- AUDIO ANALYSIS ----------------

    transcript_text = "No audio detected."

    sentiment = 0

    voice_conf = 0

    try:

        clip = VideoFileClip(filepath)

        if clip.audio is not None:

            audio_path = os.path.join(
                UPLOAD_FOLDER,
                "audio.wav"
            )

            clip.audio.write_audiofile(
                audio_path,
                verbose=False,
                logger=None
            )

            result = whisper_model.transcribe(audio_path)

            transcript_text = result["text"]

            sentiment = TextBlob(
                transcript_text
            ).sentiment.polarity

            fillers = [
                "um",
                "uh",
                "like",
                "actually",
                "basically",
                "you know"
            ]

            filler_count = sum(
                transcript_text.lower().count(word)
                for word in fillers
            )

            voice_conf = max(0, 100 - filler_count * 5)

    except Exception as e:
        print("Audio processing error:", e)

    # ---------------- FINAL SCORE ----------------

    final_score = (
        (face_score * 0.5) +
        (voice_conf * 0.3) +
        (sentiment * 20)
    )

    global latest_result

    latest_result = {
        "face_score": round(face_score, 2),
        "voice_conf": round(voice_conf, 2),
        "sentiment": round(sentiment, 2),
        "final_score": round(final_score, 2),
        "transcript": transcript_text,
        "emotions": dict(emotion_counter)
    }

    history_data.append({
        "video": filename,
        "score": round(final_score, 2)
    })

    return render_template(
        "analyze.html",
        result=latest_result
    )

# ---------------- ANALYTICS PAGE ----------------

@app.route('/analytics')
def analytics():

    if not latest_result:
        return redirect(url_for('analyze_page'))

    return render_template(
        "analytics.html",
        data=latest_result
    )

# ---------------- HISTORY PAGE ----------------

@app.route('/history')
def history():

    return render_template(
        "history.html",
        history=history_data
    )

# ---------------- RUN APP ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
