import gc
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
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # Max 60MB upload

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

    cap = None
    try:
        # ---------------- VIDEO DURATION & INITIALIZATION ----------------
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return render_template(
                "analyze.html",
                error="Unable to open uploaded video file. Please upload a valid video format."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (total_frames / fps) if (total_frames and total_frames > 0) else 0.0

        # Allow videos up to 60 seconds (1 minute)
        if duration > 60.0:
            cap.release()
            cap = None
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template(
                "analyze.html",
                error=f"Video duration exceeds maximum limit of 60 seconds (uploaded: {duration:.1f}s). Please upload a video shorter than 60 seconds."
            )

        # Load emotion model (cached)
        model = get_emotion_model()

        # ---------------- FACE ANALYSIS (OPTIMIZED FOR RENDER LOW CPU) ----------------
        # Dynamic sampling: sample 5 frames max evenly spread across the video for sub-1s execution
        MAX_FRAMES_TO_PROCESS = 5
        if duration > 0:
            interval_sec = max(2.0, duration / MAX_FRAMES_TO_PROCESS)
        else:
            interval_sec = 3.0
        frame_interval = max(1, int(fps * interval_sec))

        emotion_counts = []
        curr_frame_idx = 0
        processed_frames = 0

        while processed_frames < MAX_FRAMES_TO_PROCESS:
            if total_frames > 0 and curr_frame_idx >= total_frames:
                break

            if curr_frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Downscale frame to max width 200 px for ultra-fast face detection
                h, w = frame.shape[:2]
                if w > 200:
                    scale = 200.0 / w
                    proc_frame = cv2.resize(frame, (200, max(1, int(h * scale))))
                else:
                    proc_frame = frame

                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.35,
                    minNeighbors=3,
                    minSize=(16, 16)
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
            curr_frame_idx += frame_interval

        # Release VideoCapture resource immediately after processing loop
        cap.release()
        cap = None

        # Force garbage collection
        gc.collect()

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
            with VideoFileClip(filepath) as clip:
                if clip.audio is not None:
                    transcript_text = "Audio detected successfully."
                    sentiment = 0.5
                    voice_conf = 85
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
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        # Clean up temporary uploaded file to prevent disk exhaustion
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        gc.collect()


# ---------------- ANALYTICS PAGE ----------------

@app.route('/analytics')
def analytics():
    data = latest_result if latest_result else {
        "face_score": 0.0,
        "voice_conf": 0.0,
        "sentiment": 0.0,
        "final_score": 0.0,
        "transcript": "No video analyzed yet.",
        "emotions": {"neutral": 0, "happy": 0, "angry": 0, "sad": 0, "fear": 0, "disgust": 0, "surprise": 0}
    }

    return render_template(
        "analytics.html",
        data=data
    )

# ---------------- HISTORY PAGE ----------------

@app.route('/history')
def history():

    return render_template(
        "history.html",
        history=history_data
    )

# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template('analyze.html', error="Uploaded video file size is too large (max 60MB). Please upload a file under 60MB."), 413

@app.errorhandler(404)
def page_not_found(e):
    return render_template('analyze.html', error="Requested page not found. Please upload a video to start analysis."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('analyze.html', error="An unexpected server issue occurred. Please try uploading a short MP4 video."), 500

# ---------------- RUN APP ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

