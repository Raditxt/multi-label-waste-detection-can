# webapp/app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, Response
from werkzeug.utils import secure_filename
import os
import uuid
import sys
import cv2
import numpy as np
import threading # Tambahan: untuk mengelola threading kamera (opsional tapi disarankan)
import time # Tambahan: untuk sleep


# Tambahkan direktori 'webapp' ke Python path agar bisa import dari 'utils'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predict import init_model, predict_image_path, preprocess_image_for_model 

app = Flask(__name__)

# --- Konfigurasi Aplikasi Flask ---
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Batas ukuran file 5MB
app.secret_key = 'your_super_secret_key_here' # Ganti dengan kunci rahasia yang kuat!

# Pastikan direktori uploads ada. Jika belum, buat.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- Konfigurasi Dinamis untuk Sumber Kamera ---
# Default webcam (0), bisa diubah ke URL atau path video (misal: 'http://ip_cam/stream' atau 'video.mp4')
VIDEO_SOURCE = 0 

# Deklarasikan variabel global untuk menyimpan hasil prediksi webcam terakhir
latest_webcam_prediction_results = {"detected_labels": {"Memuat...": 0.0}} 

# --- Pemuatan Model (Dilakukan sekali saat aplikasi dimulai) ---
print("--- Memuat Model, Label, dan Threshold Optimal ---")
try:
    model, LABELS_FINAL, OPTIMAL_THRESHOLDS = init_model()
    print("Model, label, dan threshold berhasil dimuat!")
    print(f"Jumlah label yang dikenali: {len(LABELS_FINAL)}")
except Exception as e:
    print(f"ERROR: Gagal memuat model atau konfigurasi: {e}")
    print("Aplikasi akan berjalan, tetapi prediksi tidak akan berfungsi.")
    model = None
    LABELS_FINAL = []
    OPTIMAL_THRESHOLDS = {}

# --- Fungsi Bantuan untuk Validasi File ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Route Utama: Upload Gambar dan Tampilkan Hasil ---
@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    prediction_results = None

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Tidak ada bagian file "image" dalam request.')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('Tidak ada file yang dipilih.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename) 
            file.save(file_path)
            
            if model:
                print(f"Mulai prediksi untuk file: {file_path}")
                prediction_results = predict_image_path(model, file_path, LABELS_FINAL, OPTIMAL_THRESHOLDS)
                print(f"Hasil prediksi: {prediction_results}")
            else:
                flash("Model tidak dimuat. Prediksi tidak dapat dilakukan.")
                prediction_results = {"error": "Model not loaded."}
            
            image_path = url_for('static', filename='uploads/' + unique_filename)
            
        else:
            flash('Jenis file tidak diizinkan. Harap unggah gambar (png, jpg, jpeg, gif).')
            return redirect(request.url)
            
    return render_template('index.html', image_path=image_path, prediction=prediction_results)


# --- Integrasi Webcam Detection ---

# Generator Stream Frame dari Webcam dengan Prediksi On-the-Fly
def gen_frames():
    global model, LABELS_FINAL, OPTIMAL_THRESHOLDS, latest_webcam_prediction_results, VIDEO_SOURCE
    
    # Gunakan VIDEO_SOURCE global yang dapat diubah secara dinamis
    cap = cv2.VideoCapture(VIDEO_SOURCE) 
    
    # Berikan waktu singkat agar kamera inisialisasi
    time.sleep(0.5) 

    if not cap.isOpened():
        print(f"Error: Could not open video stream from source {VIDEO_SOURCE}. Please check camera connection or source URL.")
        latest_webcam_prediction_results = {"error": "Webcam tidak dapat diakses."}
        return

    frame_count = 0
    prediction_interval = 30  # Setiap 30 frame (~1 detik jika 30 FPS)

    while True:
        success, frame = cap.read()
        if not success:
            print(f"Error: Failed to read frame from webcam source {VIDEO_SOURCE}. Exiting stream.")
            latest_webcam_prediction_results = {"error": "Gagal membaca frame."}
            break
        
        # Tambahkan delay kecil untuk mengurangi penggunaan CPU/GPU pada video stream
        # Ini bisa membantu menjaga responsivitas server jika tidak melakukan prediksi setiap frame
        # time.sleep(0.01) 

        frame_count += 1

        if frame_count % prediction_interval == 0 and model:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (224, 224))
                normalized = np.array(resized_frame) / 255.0
                input_tensor = np.expand_dims(normalized, axis=0)

                predictions = model.predict(input_tensor, verbose=0)[0]

                current_frame_prediction = {}
                for i, label in enumerate(LABELS_FINAL):
                    threshold = OPTIMAL_THRESHOLDS.get(label, 0.5) 
                    if predictions[i] >= threshold:
                        current_frame_prediction[label] = float(predictions[i])
                
                if not current_frame_prediction:
                    current_frame_prediction = {"Tidak Ditemukan Sampah Spesifik": 1.0}
                
                latest_webcam_prediction_results = {"detected_labels": current_frame_prediction}
                
            except Exception as e:
                print(f"Error during prediction in webcam stream: {e}")
                latest_webcam_prediction_results = {"error": f"Error Prediksi: {e}"} 
        
        labels_to_display = latest_webcam_prediction_results.get("detected_labels", {"Memuat...": 0.0})
        if "error" in latest_webcam_prediction_results:
             labels_to_display = {"Error": 1.0}


        y_offset = 30
        for label, prob in labels_to_display.items():
            if label == "Error":
                text = f"Error: {latest_webcam_prediction_results['error']}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                text = f"{label.replace('_', ' ').title()}: {prob*100:.1f}%"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30


        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release() 

# Route: Halaman Webcam Detection
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Endpoint baru untuk mendapatkan prediksi webcam terakhir via polling
@app.route('/get_latest_webcam_prediction')
def get_latest_webcam_prediction():
    global latest_webcam_prediction_results
    return jsonify(latest_webcam_prediction_results)

# Route Stream Feed (dipanggil dari <img src="/video_feed"> di webcam.html)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Endpoint untuk Mengubah Sumber Kamera Secara Dinamis ---
@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global VIDEO_SOURCE
    data = request.get_json() # Menerima data JSON
    new_source = data.get("source") # Mendapatkan nilai 'source'

    if new_source is not None:
        try:
            # Coba konversi ke integer jika nilainya bisa diubah
            VIDEO_SOURCE = int(new_source)
        except ValueError:
            # Jika tidak bisa, biarkan sebagai string (untuk URL/path file)
            VIDEO_SOURCE = new_source
        
        print(f"Sumber video diubah menjadi: {VIDEO_SOURCE}. Stream akan otomatis di-reset.")
        # Memberi tahu front-end untuk me-reload img src agar stream baru dimulai
        return jsonify({"status": "ok", "message": f"Sumber video diubah ke: {VIDEO_SOURCE}. Mohon refresh halaman webcam jika diperlukan."}), 200
    else:
        return jsonify({"status": "error", "message": "Sumber video tidak valid atau tidak diberikan."}), 400


# --- Endpoint REST API Khusus (Untuk integrasi non-UI) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        if model:
            prediction_results = predict_image_path(model, file_path, LABELS_FINAL, OPTIMAL_THRESHOLDS)
        else:
            return jsonify({"error": "Model not loaded. Cannot perform prediction."}), 500
        
        try:
            os.remove(file_path)
            print(f"File {file_path} dihapus setelah prediksi API.")
        except OSError as e:
            print(f"Error removing file {file_path}: {e}")
            
        return jsonify(prediction_results)
    else:
        return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image."}), 400

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')