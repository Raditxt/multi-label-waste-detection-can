# 🧠 Multi-Label Trash Classifier

Deteksi jenis-jenis sampah **bercampur** dalam satu gambar menggunakan CNN dan multi-label classification. Proyek ini menggunakan **TrashNet** dan dataset **Kaggle Garbage** yang digabung, dengan augmentasi & kombinasi otomatis.

## 🔍 Fitur Utama
- Multi-label classification (bisa 2–4 sampah per gambar)
- Dataset auto-generated dengan augmentasi
- CNN model dengan sigmoid output
- Web interface dengan Flask
- Support deteksi **real-time via kamera**

## 📁 Struktur Folder
- `dataset/` – dataset terstruktur (images + labels.csv)
- `notebooks/` – preprocessing, training, evaluation
- `model/` – file model hasil training
- `webapp/` – web Flask frontend
- `scripts/` – pengolahan dataset

## 📜 License
MIT © 2025 Raditya Mulya Akbar
