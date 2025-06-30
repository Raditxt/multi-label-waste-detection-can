# 🧠 Multi-Label Trash Classifier

Klasifikasi **berbagai jenis sampah dalam satu gambar** menggunakan model CNN dan pendekatan multi-label classification. Proyek ini dirancang untuk mendeteksi **sampah campuran** (misalnya: plastik + kertas, logam + kaca) baik dari gambar statis maupun input webcam/IP camera secara real-time.

Dibangun sebagai bagian dari Proyek Akhir Semester D4 Teknik Komputer & Jaringan — Raditya Mulya Akbar, 2025.

## 🎯 Tujuan Proyek

* ✅ Membangun sistem deteksi sampah otomatis berbasis citra.
* ✅ Mengakomodasi **multi-label classification** (1 gambar bisa mengandung banyak jenis sampah).
* ✅ Menyediakan antarmuka **web interaktif** untuk upload & prediksi.
* ✅ Mendukung prediksi **real-time dari webcam atau IP camera**, dengan **sumber video yang dapat diganti secara dinamis**.
* ✅ Menyediakan endpoint **REST API** untuk integrasi eksternal.

## 🔍 Fitur Utama

* ✅ **Multi-label classification** dengan CNN + sigmoid activation
* ✅ Augmentasi & pelabelan campuran otomatis
* ✅ **Evaluasi model lengkap**: F1, AUC, confusion matrix, threshold optimasi
* ✅ **Web interface** berbasis Flask
* ✅ Deteksi **real-time** dari webcam, IP camera, atau file video
* ✅ **Ganti kamera secara dinamis** melalui antarmuka pengguna
* ✅ Endpoint REST API untuk klasifikasi berbasis gambar

## 🧪 Teknologi & Tools

| Komponen | Teknologi / Library |
|----------|-------------------|
| Bahasa Pemrograman | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Evaluasi Model | Scikit-learn |
| Web Server | Flask |
| Real-Time Feed | OpenCV |
| Frontend Web | HTML, CSS, JS (Custom) |
| Visualisasi | Matplotlib, NumPy, Pandas |

## 🧠 Arsitektur Model

* Arsitektur: Custom CNN / Transfer Learning (ResNet/EfficientNet)
* Output: Sigmoid layer untuk prediksi multi-label
* Loss Function: Binary Cross-Entropy
* Dataset: Gabungan TrashNet + Kaggle Garbage Dataset, dikustomisasi
* Preprocessing: Augmentasi, Resize, Normalisasi, Multilabel Generator

## 🌐 Web Application Features

| Fitur | Status |
|-------|--------|
| Upload gambar & prediksi label | ✅ |
| Web interface (Flask) | ✅ |
| Deteksi real-time dari webcam | ✅ |
| Deteksi dari IP Camera | ✅ |
| Ganti sumber kamera secara live | ✅ |
| REST API: `/api/predict` | ✅ |
| Polling prediksi live (webcam) | ✅ |

## 🧰 Struktur Folder

```
sampah-multilabel-ai/
├── dataset/                # Dataset, anotasi multi-label, checkpoints model
├── notebook/               # Notebook preprocessing, training, evaluasi model
├── venv_ai_clean/          # Virtual environment lokal (tidak disertakan)
├── webapp/
│   ├── app.py              # Aplikasi Flask utama
│   ├── templates/          # HTML files (index.html, webcam.html)
│   ├── static/             # JS, CSS, dan hasil upload
│   └── utils/              # predict.py, helper untuk model inference
```

## 🚀 Cara Menjalankan Aplikasi

1. **Aktifkan environment Python** (aktifkan `venv_ai_clean` atau gunakan `requirements.txt` jika tersedia)

2. **Jalankan Flask App**
   ```bash
   cd webapp
   python app.py
   ```

3. **Akses Web Interface**
   * Upload gambar: http://localhost:5000/
   * Webcam detection: http://localhost:5000/webcam

4. **Mengganti sumber kamera**
   * Gunakan formulir pada halaman `/webcam` untuk memasukkan:
     * `0` atau `1` → webcam lokal
     * `http://192.168.x.x:8080/video` → IP Camera (via IP Webcam Android)

## 📦 Contoh Endpoint API

### 🔍 Prediksi Gambar via API

```bash
POST /api/predict
Content-Type: multipart/form-data
Body: { image: <gambar.jpg> }
```

Response:
```json
{
  "detected_labels": {
    "plastic": 0.95,
    "paper": 0.88
  }
}
```

## 🧠 Hasil Evaluasi Model (ringkasan)

* F1-Score: 0.89 (rata-rata)
* AUC (macro): 0.93
* Akurasi multi-label: 86%
* Threshold optimal disesuaikan per-label

## 📌 Catatan Tambahan

* Pastikan IP laptop dan IP Camera berada di jaringan yang sama.
* Model inference dilakukan per frame tiap 1 detik (untuk efisiensi).
* Gambar tangkapan webcam juga dapat diprediksi secara manual (snapshot).

## 📜 Lisensi

MIT © 2025 Raditya Mulya Akbar
Open for educational, research, and non-commercial use.