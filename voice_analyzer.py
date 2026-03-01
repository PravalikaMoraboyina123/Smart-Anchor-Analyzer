import whisper
from moviepy.editor import VideoFileClip
from textblob import TextBlob
import os

# ==============================
# EXTRACT AUDIO FROM VIDEO
# ==============================

video_path = input("Enter video file name (example: anchor_video.mp4): ")

if not os.path.exists(video_path):
    print("Video file not found.")
    exit()

print("Extracting audio...")
clip = VideoFileClip(video_path)
clip.audio.write_audiofile("extracted_audio.wav")

# ==============================
# SPEECH TO TEXT USING WHISPER
# ==============================

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Transcribing audio...")
result = model.transcribe("extracted_audio.wav")
text = result["text"]

print("\n===== TRANSCRIPT =====")
print(text)

# ==============================
# SENTIMENT ANALYSIS
# ==============================

analysis = TextBlob(text)
sentiment = analysis.sentiment.polarity

print("\n===== SENTIMENT SCORE =====")
print("Polarity:", round(sentiment,2))

if sentiment > 0:
    print("Overall Tone: Positive")
elif sentiment < 0:
    print("Overall Tone: Negative")
else:
    print("Overall Tone: Neutral")

# ==============================
# FILLER WORD ANALYSIS
# ==============================

fillers = ["um", "uh", "like", "you know", "actually", "basically"]
filler_count = 0

lower_text = text.lower()

for word in fillers:
    filler_count += lower_text.count(word)

print("\nFiller Words Count:", filler_count)

# ==============================
# SPEECH CONFIDENCE SCORE
# ==============================

confidence_score = max(0, 100 - (filler_count * 5))

print("\nSpeech Confidence Score:", confidence_score)