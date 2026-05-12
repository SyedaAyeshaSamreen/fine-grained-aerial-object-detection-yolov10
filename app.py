from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Folder for uploaded images and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLO model
model = YOLO('best.pt')

app.secret_key = 'supersecretkey'

# In-memory user database (demo only)
users = {}

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


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    if file:

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(file_path)

        # YOLO prediction (FIXED)
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


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')


@app.route('/performance')
def performance():
    return render_template('performance.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)