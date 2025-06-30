# webapp/utils/predict.py

import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image # Menggunakan PIL (Pillow) karena lebih umum untuk Flask daripada keras.preprocessing.image

# --- KONSTANTA & PATH ---
# Menggunakan os.path.dirname(__file__) untuk membuat path relatif terhadap lokasi file predict.py
# Ini memastikan model dan thresholds ditemukan tidak peduli dari mana app.py dijalankan.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.h5')
OPTIMAL_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results', 'optimal_thresholds.json')

IMG_SIZE = (224, 224) # Ukuran gambar yang diharapkan oleh model

# --- VARIABEL GLOBAL UNTUK MODEL DAN KONFIGURASI ---
# Ini akan diisi oleh init_model() setelah dipanggil sekali.
_model = None
_labels_final = None
_optimal_thresholds = None

# --- FUNGSI UTAMA: INISIALISASI MODEL & THRESHOLDS ---
def init_model():
    """
    Memuat model TensorFlow, daftar label, dan threshold optimal dari file.
    Fungsi ini dirancang untuk dipanggil HANYA SEKALI saat aplikasi Flask dimulai
    untuk menghindari pemuatan ulang model yang memakan sumber daya.

    Mengembalikan:
        tuple: (model, list_of_labels, dict_of_optimal_thresholds)
    """
    global _model, _labels_final, _optimal_thresholds

    if _model is None: # Pastikan model hanya dimuat sekali
        print(f"Menginisialisasi model dari: {MODEL_PATH}")
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            # Optional: Pastikan model sudah terkompilasi jika diperlukan (biasanya tidak jika dimuat dari .h5)
            # _model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Model berhasil dimuat.")
        except Exception as e:
            print(f"ERROR: Gagal memuat model dari '{MODEL_PATH}'. Pastikan file ada dan formatnya benar.")
            raise RuntimeError(f"Gagal memuat model: {e}")

        print(f"Mencoba memuat threshold optimal dari: {OPTIMAL_THRESHOLDS_PATH}")
        try:
            with open(OPTIMAL_THRESHOLDS_PATH, 'r') as f:
                _optimal_thresholds = json.load(f)
            # Mengambil daftar label dari keys threshold yang dimuat
            _labels_final = list(_optimal_thresholds.keys())
            print("Threshold optimal berhasil dimuat.")
        except FileNotFoundError:
            print(f"PERINGATAN: File threshold optimal tidak ditemukan di '{OPTIMAL_THRESHOLDS_PATH}'.")
            print("Menggunakan threshold default 0.5 untuk semua label.")
            # Default labels jika file optimal_thresholds.json tidak ditemukan
            _labels_final = ['battery', 'organik', 'glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
            _optimal_thresholds = {label: 0.5 for label in _labels_final}
        except json.JSONDecodeError as e:
            print(f"ERROR: Gagal mendekode JSON dari '{OPTIMAL_THRESHOLDS_PATH}': {e}.")
            print("Menggunakan threshold default 0.5 untuk semua label.")
            _labels_final = ['battery', 'organik', 'glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
            _optimal_thresholds = {label: 0.5 for label in _labels_final}
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan tak terduga saat memuat threshold: {e}.")
            print("Menggunakan threshold default 0.5 untuk semua label.")
            _labels_final = ['battery', 'organik', 'glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
            _optimal_thresholds = {label: 0.5 for label in _labels_final}
    
    return _model, _labels_final, _optimal_thresholds

# --- FUNGSI BANTUAN: PREPROCESSING GAMBAR ---
def preprocess_image_for_model(image_path_or_bytes):
    """
    Memuat dan melakukan preprocessing pada gambar agar sesuai dengan input model.
    Mendukung input berupa path file (string) atau byte gambar (untuk streaming).

    Args:
        image_path_or_bytes (str atau bytes): Path ke file gambar atau byte gambar.

    Mengembalikan:
        numpy.ndarray: Array gambar yang siap untuk prediksi model.
    """
    if isinstance(image_path_or_bytes, str):
        # Memuat dari path file
        img = Image.open(image_path_or_bytes).convert('RGB')
    else:
        # Memuat dari bytes (misalnya dari request.files.read() atau webcam frame)
        from io import BytesIO
        img = Image.open(BytesIO(image_path_or_bytes)).convert('RGB')

    img = img.resize(IMG_SIZE) # Resize gambar ke ukuran target
    img_array = np.array(img) / 255.0  # Konversi ke array NumPy dan normalisasi
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch (1, H, W, C)
    return img_array

# --- FUNGSI UTAMA: MELAKUKAN PREDIKSI ---
def predict_image_path(model, image_path, labels_final, optimal_thresholds):
    """
    Melakukan prediksi multi-label pada gambar yang diberikan path-nya.

    Args:
        model (tf.keras.Model): Model TensorFlow yang sudah dimuat.
        image_path (str): Path ke file gambar yang akan diprediksi.
        labels_final (list): Daftar string nama label yang sesuai dengan output model.
        optimal_thresholds (dict): Dictionary {label: threshold_value} untuk setiap label.

    Mengembalikan:
        dict: Dictionary yang berisi label-label yang terdeteksi dan probabilitasnya.
              Contoh: {"detected_labels": {"plastic": 0.95, "paper": 0.82}}
              Mengembalikan {"error": "Pesan error"} jika terjadi masalah.
    """
    try:
        processed_image = preprocess_image_for_model(image_path)
        # Melakukan prediksi. [0] karena model.predict mengembalikan array of arrays (batch)
        predictions_proba = model.predict(processed_image)[0] 

        detected_labels = {}
        for i, label in enumerate(labels_final):
            # Mengambil threshold untuk label ini, default 0.5 jika tidak ditemukan
            threshold = optimal_thresholds.get(label, 0.5)
            
            # Jika probabilitas melebihi threshold, anggap terdeteksi
            if predictions_proba[i] >= threshold:
                # Konversi np.float32 ke float Python standar agar JSON serializable
                detected_labels[label] = float(predictions_proba[i]) 
        
        # Jika tidak ada label yang terdeteksi, tambahkan pesan khusus
        if not detected_labels:
            # Anda bisa mengadaptasi ini jika Anda memiliki kelas 'no_trash' atau 'unknown'
            # Atau hanya mengembalikan dictionary kosong.
            return {"detected_labels": {"Tidak Ditemukan Sampah Spesifik": 1.0}}

        return {"detected_labels": detected_labels}

    except Exception as e:
        print(f"Error saat prediksi untuk gambar '{image_path}': {e}")
        return {"error": f"Gagal memproses gambar atau melakukan prediksi: {e}"}

# --- BLOK EKSEKUSI UNTUK PENGUJIAN MANDIRI predict.py ---
# Ini hanya akan berjalan jika Anda menjalankan `python webapp/utils/predict.py`
# Berguna untuk menguji fungsi init_model dan predict_image_path secara terpisah dari Flask.
if __name__ == '__main__':
    print("--- Menjalankan Pengujian Mandiri predict.py ---")

    # Pastikan model dan optimal_thresholds.json ada untuk pengujian
    if not os.path.exists(MODEL_PATH):
        print(f"!!! PERINGATAN: Model tidak ditemukan di {MODEL_PATH}. Membuat dummy model untuk pengujian.")
        # Buat dummy direktori jika belum ada
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # Buat model Keras dummy sederhana
        dummy_model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
                                                  tf.keras.layers.Flatten(),
                                                  tf.keras.layers.Dense(8, activation='sigmoid')])
        dummy_model.save(MODEL_PATH)

    if not os.path.exists(OPTIMAL_THRESHOLDS_PATH):
        print(f"!!! PERINGATAN: File optimal_thresholds.json tidak ditemukan di {OPTIMAL_THRESHOLDS_PATH}. Membuat dummy file.")
        # Buat dummy direktori jika belum ada
        os.makedirs(os.path.dirname(OPTIMAL_THRESHOLDS_PATH), exist_ok=True)
        # Buat dummy thresholds
        dummy_thresholds_data = {
            'battery': 0.6, 'organik': 0.7, 'glass': 0.5, 'cardboard': 0.8,
            'metal': 0.75, 'paper': 0.65, 'plastic': 0.9, 'trash': 0.4
        }
        with open(OPTIMAL_THRESHOLDS_PATH, 'w') as f:
            json.dump(dummy_thresholds_data, f, indent=2)

    # Inisialisasi model, label, dan threshold
    try:
        test_model, test_labels, test_thresholds = init_model()
        print("\nInisialisasi berhasil untuk pengujian.")
    except Exception as e:
        print(f"\nPengujian gagal karena inisialisasi model: {e}")
        exit()

    # --- Persiapan Dummy Gambar untuk Prediksi ---
    dummy_image_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
    dummy_image_path_for_test = os.path.join(dummy_image_dir, 'test_dummy_image.png')
    
    os.makedirs(dummy_image_dir, exist_ok=True)
    if not os.path.exists(dummy_image_path_for_test):
        print(f"Membuat dummy gambar untuk pengujian di: {dummy_image_path_for_test}")
        # Buat gambar PNG dummy dengan ukuran yang benar
        dummy_img_data = np.random.randint(0, 256, size=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        Image.fromarray(dummy_img_data).save(dummy_image_path_for_test)
    else:
        print(f"Menggunakan dummy gambar yang sudah ada di: {dummy_image_path_for_test}")

    # --- Lakukan Prediksi Pengujian ---
    if os.path.exists(dummy_image_path_for_test):
        print(f"\nMelakukan prediksi pada dummy gambar: {dummy_image_path_for_test}")
        test_results = predict_image_path(test_model, dummy_image_path_for_test, test_labels, test_thresholds)
        print("\nHasil Prediksi Pengujian:")
        print(json.dumps(test_results, indent=2))
    else:
        print("\nTidak dapat melakukan pengujian prediksi karena dummy gambar tidak tersedia.")

    print("\n--- Pengujian Mandiri predict.py Selesai ---")