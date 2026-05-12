from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
import os
import subprocess
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Upload and results folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load YOLO model


# Temporary in-memory user storage
users = {}

# ---------------- HOME ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/category')
def category():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

# ---------------- REGISTER ---------------- #

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        # Validation
        if not username or not password:
            flash("Please fill all fields!", "danger")
            return redirect(url_for('register'))

        # Check existing user
        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        # Save user
        users[username] = password

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# ---------------- LOGIN ---------------- #

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        # Check credentials
        if username in users and users[username] == password:

            flash("Login successful!", "success")
            return redirect(url_for('index'))

        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

# ---------------- FILE UPLOAD ---------------- #

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        flash("No file selected!", "danger")
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        flash("No file selected!", "danger")
        return redirect(url_for('home'))

    if file:

        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(file_path)

        # ---------------- VIDEO ---------------- #

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):

            result_video_path = process_video(file_path, filename)

            return render_template(
                'result.html',
                original_file=filename,
                result_video=result_video_path
            )

        # ---------------- IMAGE ---------------- #

        else:
	    model = YOLO('best.pt')
            results = model(file_path)

            result_img_path = os.path.join(
                app.config['RESULTS_FOLDER'],
                f"result_{filename}"
            )

            annotated_frame = results[0].plot()

            cv2.imwrite(result_img_path, annotated_frame)

            return render_template(
                'result.html',
                original_file=filename,
                result_file=f"result_{filename}"
            )

# ---------------- FILE ROUTES ---------------- #

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(
        app.config['RESULTS_FOLDER'],
        filename,
        mimetype='video/mp4'
    )

# ---------------- VIDEO PROCESSING ---------------- #

def process_video(input_path, filename):
    model = YOLO('best.pt')
    cap = cv2.VideoCapture(input_path)

    output_path = os.path.join(
        app.config['RESULTS_FOLDER'],
        f"result_{filename}"
    )

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_size = (width, height)

    # Temporary video
    temp_output = os.path.join(
        app.config['RESULTS_FOLDER'],
        "temp_video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        temp_output,
        fourcc,
        fps,
        frame_size
    )

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO detection
        results = model(frame)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg conversion
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    # Remove temp file
    if os.path.exists(temp_output):
        os.remove(temp_output)

    return f"result_{filename}"

# ---------------- MAIN ---------------- #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)