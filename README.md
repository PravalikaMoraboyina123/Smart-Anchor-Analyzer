# Smart Anchor Analyzer

Smart Anchor Analyzer is an AI-powered facial emotion and communication analysis system developed using Flask, OpenCV, TensorFlow, and Machine Learning techniques. The application analyzes uploaded video files to detect facial emotions, evaluate confidence levels, and generate presentation analytics through computer vision and intelligent processing.

The project focuses on Artificial Intelligence, Computer Vision, Emotion Detection, and Web Application Development. It provides an interactive platform where users can upload videos and receive analytical insights related to facial emotions and communication behavior.

---

## Live Demo

Website: https://smart-anchor-analyzer.onrender.com/

---

## GitHub Repository

Repository Link:  
https://github.com/PravalikaMoraboyina123/Smart-Anchor-Analyzer

---

## Features

- Video upload and analysis system
- Facial emotion detection using Deep Learning
- Real-time frame processing
- Emotion classification and analytics
- Confidence and stress score calculation
- Interactive analytics dashboard
- History tracking system
- Professional Flask web application
- Responsive UI design
- Machine Learning-based prediction system
- Cloud deployment using Render

---

## Technologies Used

### Programming Language
- Python

### Frameworks & Libraries
- Flask
- OpenCV
- TensorFlow
- Keras
- NumPy
- MoviePy
- TextBlob
- Scikit-learn
- ImageIO
- Gunicorn

### Machine Learning & AI
- Deep Learning
- Facial Emotion Recognition
- Computer Vision
- Classification Models
- Image Processing

### Development Tools
- VS Code
- GitHub
- Render
- Jupyter Notebook

---

## Project Structure

```bash
Smart-Anchor-Analyzer/
│
├── app.py
├── emotion_model.h5
├── requirements.txt
├── runtime.txt
├── Procfile
├── uploads/
├── static/
├── templates/
│   ├── home.html
│   ├── analyze.html
│   ├── analytics.html
│   └── history.html
├── dataset/
├── render.yaml
└── README.md
```

---

## How the System Works

1. User uploads a video file through the web interface.
2. The system processes video frames using OpenCV.
3. Faces are detected from extracted frames.
4. Facial images are preprocessed and resized.
5. Deep Learning model predicts emotions from facial expressions.
6. Emotion statistics are generated.
7. Confidence and stress levels are calculated.
8. Final analytics and prediction scores are displayed.

---

## Emotion Categories

The system can identify emotions such as:

- Happy
- Sad
- Angry
- Neutral
- Fear
- Surprise
- Disgust

---

## Machine Learning Workflow

### Data Collection
- Facial emotion datasets were collected for training and testing.

### Data Preprocessing
- Frame extraction
- Face detection
- Image resizing
- Grayscale conversion
- Normalization

### Model Training
- Deep Learning emotion classification model was trained using TensorFlow and Keras.

### Model Evaluation
- Prediction accuracy analysis
- Emotion classification validation
- Performance testing

### Deployment
- Flask application deployed publicly using Render cloud platform.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/PravalikaMoraboyina123/Smart-Anchor-Analyzer.git
```

### Navigate to Project Directory

```bash
cd Smart-Anchor-Analyzer
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Using Flask

```bash
python app.py
```

### Production Deployment

```bash
gunicorn --workers 1 --timeout 120 app:app
```

---

## Deployment

The project is deployed publicly using Render.

### Deployment Platform
- Render Cloud Platform

### Deployment Features
- Public hosting
- Automatic GitHub deployment
- Cloud-based execution
- Production-ready Flask setup
- Gunicorn production server integration

---

## Dashboard Features

- Emotion analytics
- Confidence score display
- Stress level analysis
- Video processing system
- Prediction history
- Interactive result visualization

---

## User Interface

The application includes:
- Responsive web interface
- Clean dashboard layout
- Upload-based workflow
- Analytics visualization
- Modern Flask frontend design

---

## Optimization & Deployment Improvements

The application was optimized for cloud deployment by:
- Reducing memory usage
- Using lightweight processing methods
- Optimizing TensorFlow loading
- Configuring Render-compatible deployment
- Adding Gunicorn support
- Using OpenCV headless version for servers
- Dynamic PORT binding for Render deployment

---

## Future Enhancements

- Real-time webcam emotion detection
- Speech and voice emotion analysis
- Advanced analytics dashboard
- Multi-face emotion detection
- Deep Learning model improvements
- Cloud storage integration
- User authentication system
- AI-powered presentation feedback

## Author

Pravalika Moraboyina

B.Tech - Computer Science & Engineering (Data Science)

Rajeev Gandhi Memorial College of Engineering & Technology

GitHub: https://github.com/PravalikaMoraboyina123

Website: https://smart-anchor-analyzer.onrender.com/

Email: pravalikamoraboyina123@gmail.com

---

## License

This project is developed for educational and learning purposes.
