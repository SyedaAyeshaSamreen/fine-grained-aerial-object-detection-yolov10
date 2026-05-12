from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
import os
import cv2
import subprocess
from werkzeug.utils import secure_filename

# ---------------- APP INIT ---------------- #
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

app.secret_key = 'supersecretkey'

# ---------------- MODEL ---------------- #
model = YOLO('best.pt')

# ---------------- USERS (demo only) ---------------- #
users = {}

# ---------------- ROUTES ---------------- #

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


# ---------------- REGISTER ---------------- #
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        users[username] = password
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# ---------------- LOGIN ---------------- #
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


# ---------------- UPLOAD (IMAGE + VIDEO FIXED) ---------------- #
@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # ---------------- VIDEO ---------------- #
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return process_video(file_path, filename)

    # ---------------- IMAGE ---------------- #
    results = model.predict(file_path, imgsz=320)

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


# ---------------- FILE SERVING ---------------- #
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

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp.mp4")
    final_output = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg re-encode
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        final_output
    ]

    subprocess.run(cmd, check=True)
    os.remove(temp_output)

    return render_template(
        'result.html',
        original_file=filename,
        result_video=f"result_{filename}"
    )


# ---------------- OTHER PAGES ---------------- #
@app.route('/performance')
def performance():
    return render_template('performance.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


# ---------------- RUN ---------------- #
if __name__ == '__main__':
    app.run(debug=True, port=5002)