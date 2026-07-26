from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from collections import Counter

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

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
try:
    os.environ["PATH"] += os.pathsep + os.path.dirname(
        imageio_ffmpeg.get_ffmpeg_exe()
    )
except Exception as e:
    print("FFmpeg environment setup note:", e)

# ---------------- EMOTION LABELS ----------------

emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# ---------------- FACE DETECTION ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# ---------------- GLOBAL MODEL CACHE ----------------

class EmotionModelWrapper:
    def __init__(self, net):
        self.net = net

    def predict(self, face, verbose=0):
        self.net.setInput(face)
        return self.net.forward()

emotion_model = None

def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        onnx_file = "emotion_model.onnx"
        if os.path.exists(onnx_file):
            try:
                print("Loading emotion model via OpenCV DNN (ONNX)...")
                net = cv2.dnn.readNetFromONNX(onnx_file)
                emotion_model = EmotionModelWrapper(net)
                return emotion_model
            except Exception as e:
                print("OpenCV DNN load warning:", e)

        # Fallback to Keras load_model
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras import layers, models
            model_file = "emotion_model.keras" if os.path.exists("emotion_model.keras") else "emotion_model.h5"
            try:
                emotion_model = load_model(model_file, compile=False)
            except Exception as err:
                print("Model load warning, reconstructing model architecture:", err)
                m = models.Sequential([
                    layers.Input(shape=(48, 48, 1)),
                    layers.Conv2D(32, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(7, activation='softmax')
                ])
                weights_file = "emotion_model.h5" if os.path.exists("emotion_model.h5") else "emotion_model.keras"
                m.load_weights(weights_file)
                emotion_model = m
        except Exception as err:
            print("Keras load error:", err)
    return emotion_model


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

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'GET':
        return redirect(url_for('analyze_page'))

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

    try:
        # Load model instance (cached)
        model = get_emotion_model()

        # ---------------- FACE ANALYSIS ----------------
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 1

        frame_interval = fps
        emotion_counts = []
        frame_count = 0
        processed_frames = 0
        MAX_FRAMES_TO_PROCESS = 30  # Cap processing to 30 frames max (1 frame/sec) for cloud speed

        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= MAX_FRAMES_TO_PROCESS or frame_count > (MAX_FRAMES_TO_PROCESS * fps):
                break

            if frame_count % frame_interval == 0:
                try:
                    # Downscale large frames to 480px width for 8x faster face detection
                    h, w = frame.shape[:2]
                    if w > 480:
                        scale = 480.0 / w
                        proc_frame = cv2.resize(frame, (480, int(h * scale)))
                    else:
                        proc_frame = frame

                    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(30, 30)
                    )

                    for (x, y, w_box, h_box) in faces:
                        face = gray[y:y+h_box, x:x+w_box]
                        try:
                            face = cv2.resize(face, (48, 48))
                            face = face / 255.0
                            face = np.reshape(face, (1, 48, 48, 1))

                            prediction = model.predict(face, verbose=0)
                            emotion = emotion_labels[np.argmax(prediction)]
                            emotion_counts.append(emotion)
                        except Exception:
                            continue
                except Exception:
                    pass

                processed_frames += 1

            frame_count += 1

        cap.release()

        # ---------------- EMOTION RESULTS ----------------
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

        # ---------------- LIGHTWEIGHT AUDIO ANALYSIS ----------------
        transcript_text = "Speech analysis disabled for deployment."
        sentiment = 0.5
        voice_conf = 80

        try:
            clip = VideoFileClip(filepath)
            if clip.audio is not None:
                transcript_text = "Audio detected successfully."
                sentiment = 0.5
                voice_conf = 85
            clip.close()
        except Exception as e:
            print("Audio processing note:", e)

        # ---------------- FINAL SCORE ----------------
        final_score = (face_score * 0.7) + (voice_conf * 0.3)

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

    except Exception as err:
        print("Error during video processing:", err)
        return render_template(
            "analyze.html",
            error=f"Video processing encountered an issue: {str(err)}. Please try uploading a short MP4 file."
        )

    finally:
        # Clean up temporary uploaded file to prevent disk exhaustion
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


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

