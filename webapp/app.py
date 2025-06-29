import os
import random
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from sklearn.model_selection import train_test_split # Import untuk split data
import numpy as np # Import untuk statistik

# ===== KONFIGURASI =====
LABELS_FINAL = ['battery', 'organik', 'glass', 'cardboard', 'metal', 'paper', 'plastic', 'trash']
LABEL_MAP = {
    # Kaggle
    'battery': 'battery',
    'biological': 'organik',
    'brown-glass': 'glass',
    'white-glass': 'glass',
    'green-glass': 'glass',
    'clothes': 'trash',
    'shoes': 'trash',
    'metal': 'metal',
    'paper': 'paper',
    'cardboard': 'cardboard',
    'plastic': 'plastic',
    'trash': 'trash',
    # TrashNet
    'glass': 'glass'
}

# Konfigurasi Tambahan untuk Dataset Generation
NUM_IMAGES_TO_GENERATE = 5000 # Jumlah total gambar gabungan yang akan dibuat
MIN_LABELS_PER_IMAGE = 2     # Minimum label (objek) per gambar gabungan
MAX_LABELS_PER_IMAGE = 4     # Maksimum label (objek) per gambar gabungan
TARGET_IMG_SIZE = (224, 224) # Ukuran target untuk setiap objek individual sebelum digabungkan
TRAIN_TEST_SPLIT_RATIO = 0.2 # Rasio untuk test/validation set (0.2 = 20% untuk validasi)
RANDOM_STATE_SPLIT = 42      # Seed untuk reproduksibilitas split

# ===== PATH =====
BASE_DIR = 'dataset'
KAGGLE_PATH = os.path.join(BASE_DIR, 'original_kaggle')
TRASHNET_PATH = os.path.join(BASE_DIR, 'original_trashnet')
OUTPUT_IMG_PATH = os.path.join(BASE_DIR, 'images')
LABELS_CSV_PATH = os.path.join(BASE_DIR, 'labels.csv')
TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'train.csv')
VAL_CSV_PATH = os.path.join(BASE_DIR, 'val.csv')
ERROR_LOG_PATH = os.path.join(BASE_DIR, 'error.log') # Path untuk log error

os.makedirs(OUTPUT_IMG_PATH, exist_ok=True)

# Kosongkan file error log setiap kali skrip dijalankan
if os.path.exists(ERROR_LOG_PATH):
    os.remove(ERROR_LOG_PATH)
# Buat direktori dataset jika belum ada, untuk memastikan error.log bisa ditulis
os.makedirs(BASE_DIR, exist_ok=True)


# ===== FUNGSI BANTUAN =====

def apply_random_augmentation(img):
    """Menerapkan augmentasi dasar secara acak pada gambar."""
    # Random Horizontal Flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Random Rotation (hanya 0, 90, 180, 270 derajat untuk menghindari padding tambahan)
    rotations = [0, 90, 180, 270]
    img = img.rotate(random.choice(rotations), expand=True)

    # Random Color Jitter (contoh sederhana)
    if random.random() < 0.3:
        r, g, b = img.split()
        r = r.point(lambda i: i * random.uniform(0.8, 1.2))
        g = g.point(lambda i: i * random.uniform(0.8, 1.2))
        b = b.point(lambda i: i * random.uniform(0.8, 1.2))
        img = Image.merge('RGB', (r, g, b))

    return img

def resize_with_padding(img, target_size=TARGET_IMG_SIZE, color=(0, 0, 0)):
    img = img.convert('RGB')
    old_size = img.size
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, color)
    paste_pos = ((target_size[0] - new_size[0]) // 2,
                 (target_size[1] - new_size[1]) // 2)
    new_img.paste(img, paste_pos)
    return new_img

def get_all_class_paths():
    paths = {}
    for root_path in [KAGGLE_PATH, TRASHNET_PATH]:
        if not os.path.exists(root_path):
            print(f"Peringatan: Direktori '{root_path}' tidak ditemukan. Melewatkan.")
            continue
        for folder in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder)
            if os.path.isdir(folder_path):
                if folder in LABEL_MAP:
                    label_final = LABEL_MAP[folder]
                    # Pastikan folder memiliki setidaknya satu gambar
                    if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for file in os.listdir(folder_path)):
                        paths.setdefault(label_final, []).append(folder_path)
                    else:
                        print(f"Peringatan: Folder '{folder_path}' kosong atau tidak mengandung gambar yang didukung. Melewatkan.")
                # else: # Ini akan sangat verbose jika banyak folder yang tidak ada di LABEL_MAP
                #     print(f"Peringatan: Folder '{folder_path}' tidak ada di LABEL_MAP. Melewatkan.")
    return paths

def get_random_image_path(label_class, class_paths):
    if label_class not in class_paths or not class_paths[label_class]:
        raise ValueError(f"Tidak ada path folder yang tersedia untuk kelas label: {label_class}")

    chosen_folder = random.choice(class_paths[label_class])
    
    # Filter hanya file gambar
    image_files = [f for f in os.listdir(chosen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not image_files:
        raise ValueError(f"Folder '{chosen_folder}' tidak mengandung gambar yang didukung untuk kelas: {label_class}")

    chosen_file = random.choice(image_files)
    return os.path.join(chosen_folder, chosen_file)

# ===== PROSES UTAMA =====
class_paths = get_all_class_paths()

# Verifikasi bahwa ada cukup data di semua kelas yang diinginkan
for lbl in LABELS_FINAL:
    if lbl not in class_paths or not class_paths[lbl]:
        print(f"Error: Tidak ada data sumber yang ditemukan untuk label '{lbl}'. Harap periksa path dan struktur folder Anda.")
        exit() # Keluar jika ada label yang tidak memiliki data

data = []

print("Memulai pembuatan dataset...")
# Membuka file log error dalam mode 'append'
with open(ERROR_LOG_PATH, "a") as logf:
    for i in range(NUM_IMAGES_TO_GENERATE):
        try:
            # Menentukan jumlah label yang akan dipilih untuk gambar ini
            num_selected_labels = random.randint(MIN_LABELS_PER_IMAGE, MAX_LABELS_PER_IMAGE)
            
            # Memilih label secara acak, pastikan label yang dipilih unik
            # Menambahkan safety check jika LABELS_FINAL lebih kecil dari num_selected_labels
            if num_selected_labels > len(LABELS_FINAL):
                num_selected_labels = len(LABELS_FINAL)
            selected_labels = random.sample(LABELS_FINAL, num_selected_labels)
            
            # Mengambil path gambar untuk label yang dipilih
            selected_imgs_paths = [get_random_image_path(lbl, class_paths) for lbl in selected_labels]
            
            # Membuka gambar dan menerapkan augmentasi sebelum resize & padding
            images_to_combine = []
            for p in selected_imgs_paths:
                img = Image.open(p)
                img = apply_random_augmentation(img) # Terapkan augmentasi
                images_to_combine.append(resize_with_padding(img, target_size=TARGET_IMG_SIZE))
                
            # Acak urutan gambar sebelum digabungkan
            random.shuffle(images_to_combine)

            # Menggabungkan gambar
            total_width = TARGET_IMG_SIZE[0] * len(images_to_combine)
            combined = Image.new('RGB', (total_width, TARGET_IMG_SIZE[1]))
            for idx, img in enumerate(images_to_combine):
                combined.paste(img, (idx * TARGET_IMG_SIZE[0], 0))

            # Format angka dinamis
            filename = f"img_{i:0{len(str(NUM_IMAGES_TO_GENERATE - 1))}}.jpg"
            combined.save(os.path.join(OUTPUT_IMG_PATH, filename))

            label_row = [filename] + [1 if lbl in selected_labels else 0 for lbl in LABELS_FINAL]
            data.append(label_row)

        except Exception as e:
            error_message = f"Error saat memproses iterasi {i}: {e}"
            print(f"{error_message}. Melewatkan gambar ini.")
            logf.write(f"[{i}] {e}\n") # Simpan error ke file log
            continue # Lanjutkan ke iterasi berikutnya

print(f"Selesai: {len(data)} gambar berhasil dibuat di {OUTPUT_IMG_PATH}")
if os.path.getsize(ERROR_LOG_PATH) > 0:
    print(f"Periksa {ERROR_LOG_PATH} untuk daftar error yang dilewati.")
else:
    print("Tidak ada error yang dicatat dalam proses ini.")

# Simpan CSV
if data: # Pastikan ada data sebelum menyimpan
    columns = ['filename'] + LABELS_FINAL
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(LABELS_CSV_PATH, index=False)
    print(f"File label keseluruhan disimpan di {LABELS_CSV_PATH}")

    # ===== SPLIT OTOMATIS (TRAIN/VALIDATION) =====
    print("Melakukan split dataset train/validation...")
    train_df, val_df = train_test_split(df, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE_SPLIT)
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)
    print(f"Dataset train disimpan di {TRAIN_CSV_PATH} ({len(train_df)} sampel)")
    print(f"Dataset validation disimpan di {VAL_CSV_PATH} ({len(val_df)} sampel)")

    # ===== STATISTIK DATASET =====
    print("\nStatistik Kemunculan Label di Dataset Keseluruhan:")
    # Pastikan label_array dibuat dari df yang sudah ada
    # Drop kolom 'filename' dan konversi ke numpy array
    label_counts = df[LABELS_FINAL].sum(axis=0) 
    for label, count in label_counts.items():
        print(f"  {label:<10}: {int(count)} muncul")

else:
    print("Tidak ada gambar yang berhasil dibuat. Tidak ada CSV yang disimpan.")