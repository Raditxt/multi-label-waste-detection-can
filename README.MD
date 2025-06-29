# 🧠 Multi-Label Trash Classifier

Klasifikasi **berbagai jenis sampah** dalam satu gambar menggunakan CNN dan pendekatan multi-label classification. Proyek ini dirancang untuk mengenali **sampah campuran** seperti plastik + kertas, logam + kaca, dll.

Model dilatih dari kombinasi dataset **TrashNet** dan **Kaggle Garbage Classification**, dengan augmentasi otomatis untuk menciptakan sampel multi-label.

---

## 🔍 Fitur Utama

- ✅ Multi-label classification (bisa lebih dari 1 jenis sampah per gambar)
- 🔁 Augmentasi otomatis + pelabelan campuran  
- 📊 Evaluasi lengkap (F1, AUC, confusion matrix, threshold optimal)
- 🧠 CNN custom dengan **sigmoid output**
- 🌐 Web interface berbasis Flask
- 🎥 Deteksi **real-time via webcam** (📦 dalam pengembangan)

---

## 🧪 Teknologi yang Digunakan

- **Python** 3.10+
- **TensorFlow / Keras**
- **scikit-learn** untuk evaluasi
- **Flask** untuk web deployment
- **OpenCV** (untuk webcam mode, next step)
- **Matplotlib / NumPy / Pandas** (visualisasi & handling data)

---

## 📁 Struktur Folder

```
sampah-multilabel-ai/
├── dataset/          # Gambar asli, multi-label CSV, model checkpoints
├── notebook/         # Notebook: preprocessing, training, evaluasi model
├── venv_ai_clean/    # Virtual environment (tidak disertakan versi upload)
├── webapp/           # Aplikasi Flask untuk prediksi gambar & webcam
```

---

## 🚀 Deployment

- ✅ Upload gambar untuk klasifikasi melalui webapp
- 🕐 Deteksi real-time via webcam: **dalam tahap integrasi**

---

## 📜 License

MIT © 2025 Raditya Mulya Akbar