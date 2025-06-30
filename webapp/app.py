# webapp/app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
import os
import uuid
import sys

# Tambahkan direktori 'webapp' ke Python path agar bisa import dari 'utils'
# Ini penting agar Flask bisa menemukan modul 'utils.predict'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predict import init_model, predict_image_path

app = Flask(__name__)

# --- Konfigurasi Aplikasi Flask ---
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Batas ukuran file 5MB
app.secret_key = 'your_super_secret_key_here' # Ganti dengan kunci rahasia yang kuat! Diperlukan untuk flash messages

# Pastikan direktori uploads ada. Jika belum, buat.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Pemuatan Model (Dilakukan sekali saat aplikasi dimulai) ---
# Ini adalah bagian krusial untuk efisiensi. Model, label, dan threshold dimuat hanya satu kali.
print("--- Memuat Model, Label, dan Threshold Optimal ---")
try:
    model, LABELS_FINAL, OPTIMAL_THRESHOLDS = init_model()
    print("Model, label, dan threshold berhasil dimuat!")
    print(f"Jumlah label yang dikenali: {len(LABELS_FINAL)}")
    # print(f"Optimal Thresholds: {OPTIMAL_THRESHOLDS}") # Bisa di-uncomment untuk debugging
except Exception as e:
    print(f"ERROR: Gagal memuat model atau konfigurasi: {e}")
    print("Aplikasi akan berjalan, tetapi prediksi tidak akan berfungsi.")
    model = None # Set model ke None agar tidak crash jika terjadi kegagalan
    LABELS_FINAL = []
    OPTIMAL_THRESHOLDS = {}

# --- Fungsi Bantuan untuk Validasi File ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    """
    Memeriksa apakah ekstensi file diizinkan.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Route Utama: Upload Gambar dan Tampilkan Hasil ---
@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    prediction_results = None # Mengubah nama variabel agar lebih deskriptif

    if request.method == 'POST':
        # 1. Periksa apakah ada file dalam request
        if 'image' not in request.files:
            flash('Tidak ada bagian file "image" dalam request.')
            return redirect(request.url)

        file = request.files['image']

        # 2. Periksa apakah pengguna memilih file
        if file.filename == '':
            flash('Tidak ada file yang dipilih.')
            return redirect(request.url)
        
        # 3. Validasi file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Buat nama file unik menggunakan UUID untuk menghindari konflik
            unique_filename = str(uuid.uuid4()) + "_" + filename
            
            # Mendapatkan path lengkap untuk menyimpan file
            # os.path.join() di sini tetap benar karena ini untuk path sistem file lokal
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename) 
            file.save(file_path)
            
            # Lakukan prediksi jika model berhasil dimuat
            if model:
                print(f"Mulai prediksi untuk file: {file_path}")
                prediction_results = predict_image_path(model, file_path, LABELS_FINAL, OPTIMAL_THRESHOLDS)
                print(f"Hasil prediksi: {prediction_results}")
            else:
                flash("Model tidak dimuat. Prediksi tidak dapat dilakukan.")
                prediction_results = {"error": "Model not loaded."}
            
            # --- Perbaikan: Menggunakan string concatenation dengan '/' untuk URL ---
            # Ini akan memastikan URL yang benar (menggunakan forward slashes)
            image_path = url_for('static', filename='uploads/' + unique_filename)
            # --- Akhir Perbaikan ---
            
            # Opsional: Hapus file setelah prediksi. Untuk debugging, mungkin ingin disimpan.
            # Jika Anda tidak menghapusnya, pastikan ada mekanisme pembersihan file lama.
            # os.remove(file_path)
            # print(f"File {file_path} dihapus setelah prediksi.")
            
        else:
            flash('Jenis file tidak diizinkan. Harap unggah gambar (png, jpg, jpeg, gif).')
            return redirect(request.url)
            
    # Render template dengan path gambar dan hasil prediksi
    return render_template('index.html', image_path=image_path, prediction=prediction_results)

# --- Endpoint REST API Khusus (Untuk integrasi non-UI) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # 1. Periksa apakah ada file dalam request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # 2. Periksa apakah pengguna memilih file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 3. Validasi file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Lakukan prediksi jika model berhasil dimuat
        if model:
            prediction_results = predict_image_path(model, file_path, LABELS_FINAL, OPTIMAL_THRESHOLDS)
        else:
            return jsonify({"error": "Model not loaded. Cannot perform prediction."}), 500
        
        # Selalu hapus file yang diunggah untuk API setelah prediksi, untuk menjaga kebersihan server.
        try:
            os.remove(file_path)
            print(f"File {file_path} dihapus setelah prediksi API.")
        except OSError as e:
            print(f"Error removing file {file_path}: {e}") # Log error jika gagal hapus
        
        return jsonify(prediction_results)
    else:
        return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image."}), 400

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    # Pastikan direktori UPLOAD_FOLDER ada saat startup
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Untuk pengembangan, gunakan debug=True.
    # Untuk deployment produksi, set debug=False dan gunakan WSGI server (misal Gunicorn).
    # host='0.0.0.0' akan membuat aplikasi dapat diakses dari luar localhost (misal dari jaringan lokal Anda).
    app.run(debug=True, host='0.0.0.0')