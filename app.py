"""
Emotion Detection Web Application
Flask backend for facial emotion recognition
"""

import os
import cv2
import pickle
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from PIL import Image
import io
import sqlite3
from model import EMOTIONS, build_emotion_model, compile_model

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model and cascade classifier
try:
    model = load_model('emotion_model.h5')
    print("✓ Emotion model loaded successfully")
except:
    print("⚠ Creating new model...")
    model = build_emotion_model()
    model = compile_model(model)
    model.save('emotion_model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Database setup
def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            name TEXT,
            emotion TEXT,
            confidence REAL,
            image_path TEXT,
            detection_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(name, emotion, confidence, image_path, detection_type):
    """Save detection result to database."""
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (name, emotion, confidence, image_path, detection_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, emotion, confidence, image_path, detection_type, datetime.now()))
    conn.commit()
    conn.close()

def get_all_detections():
    """Retrieve all detection records."""
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY timestamp DESC')
    detections = c.fetchall()
    conn.close()
    return detections

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_face(face_roi):
    """Preprocess face region for emotion detection."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=-1)
    return gray

def detect_emotion_image(image_path):
    """Detect emotion from uploaded image."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    results = []
    
    if len(faces) == 0:
        return None, "No faces detected in image"
    
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        processed_roi = preprocess_face(roi)
        
        # Predict emotion
        emotion_pred = model.predict(processed_roi, verbose=0)
        emotion_idx = np.argmax(emotion_pred)
        emotion = EMOTIONS[emotion_idx]
        confidence = float(emotion_pred[0][emotion_idx]) * 100
        
        results.append({
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    
    return results, None

# Routes
@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and emotion detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        name = request.form.get('name', 'Anonymous')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Detect emotion
        results, error = detect_emotion_image(filepath)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Save to database
        if results:
            primary_emotion = results[0]['emotion']
            confidence = results[0]['confidence']
            save_detection(name, primary_emotion, confidence, filename, 'image')
        
        return jsonify({
            'success': True,
            'results': results,
            'image_path': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/webcam', methods=['POST'])
def webcam_detection():
    """Handle real-time webcam emotion detection."""
    try:
        data = request.json
        image_data = data.get('image')
        name = data.get('name', 'Anonymous')
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = f"{timestamp}webcam.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect emotion
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        results = []
        for (x, y, w, h) in faces:
            roi = cv_image[y:y+h, x:x+w]
            processed_roi = preprocess_face(roi)
            
            emotion_pred = model.predict(processed_roi, verbose=0)
            emotion_idx = np.argmax(emotion_pred)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(emotion_pred[0][emotion_idx]) * 100
            
            results.append({
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        # Save to database
        if results:
            save_detection(name, results[0]['emotion'], results[0]['confidence'], filename, 'webcam')
        
        return jsonify({
            'success': True,
            'results': results,
            'image_path': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get detection history."""
    detections = get_all_detections()
    return jsonify({
        'success': True,
        'count': len(detections),
        'detections': [
            {
                'id': d[0],
                'name': d[1],
                'emotion': d[2],
                'confidence': d[3],
                'type': d[5],
                'timestamp': d[6]
            }
            for d in detections
        ]
    })

@app.route('/stats')
def get_stats():
    """Get emotion detection statistics."""
    detections = get_all_detections()
    emotion_counts = {}
    
    for d in detections:
        emotion = d[2]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return jsonify({
        'success': True,
        'total_detections': len(detections),
        'emotion_distribution': emotion_counts
    })

@app.route('/image/<filename>')
def get_image(filename):
    """Serve uploaded image."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)