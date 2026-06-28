# Klasifikasi Sampah Rumah Tangga dengan Model EfficientNetB0 menggunakan Transfer Learning dan Data Augmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muhammadfathanfuad/household-waste-deeplearning/blob/main/klasifikasi_sampah_rumah_tangga_dengan_Model_EfficientNetB0_menggunakan_Transfer_Learning_dan_Data_Augmentation.ipynb)

Proyek ini berfokus pada pengembangan model Deep Learning berbasis computer vision untuk mengklasifikasikan jenis sampah rumah tangga ke dalam **50 kelas kategori**. Menggunakan arsitektur state-of-the-art **EfficientNetB0** melalui teknik **Transfer Learning** dan **Data Augmentation**, model ini dirancang untuk mendeteksi sampah secara presisi dan efisien guna mempermudah proses pemilahan sampah otomatis di lingkungan rumah tangga.

---

## 📌 Fitur Utama Proyek
- **50 Kelas Klasifikasi**: Mendukung klasifikasi detail dari berbagai macam sampah anorganik (plastik, kaca, kertas, logam, dll.) dan organik (sisa makanan, kulit buah, dll.).
- **EfficientNetB0 Backbone**: Menggunakan bobot pretrained dari *ImageNet* untuk performa ekstraksi fitur yang unggul dengan parameter yang relatif ringan (~4.1 Juta parameter).
- **Two-Phase Training**:
  - **Phase 1: Feature Extraction** untuk melatih klasifikasi top-layer dengan basis model ter-freeze.
  - **Phase 2: Fine-Tuning** untuk melatih 20 layer teratas base model dengan learning rate sangat kecil demi penyesuaian fitur tingkat tinggi.
- **Data Augmentation Terintegrasi**: Menggunakan Keras Preprocessing Layers langsung di dalam model (`Sequential`) untuk mencegah overfitting selama pelatihan.
- **Visualisasi Komprehensif**: Menyediakan visualisasi distribusi dataset asli, resolusi sebaran gambar, matriks kebingungan (confusion matrix) ternormalisasi, performa F1-Score per kelas, serta hasil uji prediksi sampel.
- **Inference Ready**: Dilengkapi fungsi interaktif untuk langsung mengunggah dan menguji gambar sampah baru di lingkungan Google Colab.

---

## 📊 Dataset & Distribusi Kelas
Dataset ini terdiri dari **2.511 gambar** yang tersebar di **50 kelas**. Rata-rata setiap kelas memiliki 50 gambar, dengan sedikit variasi pada beberapa kelas.

### Statistik Dataset:
- **Total Gambar**: 2.511 gambar
- **Jumlah Kelas**: 50 kelas
- **Rata-rata Gambar Per Kelas**: 50,2 gambar
- **Resolusi Target Model**: 224 x 224 piksel (RGB)

### Distribusi Gambar Per Kelas:
| No | Nama Kelas | Jumlah Gambar |
|---|---|---|
| 1 | gelas_plastik_bening | 54 |
| 2 | kulit_pisang | 54 |
| 3 | pecahan_kaca | 51 |
| 4 | sisa_sayuran | 51 |
| 5 | struk_belanja | 51 |
| 6 | Styrofoam | 50 |
| 7 | ampas_kopi_teh | 50 |
| 8 | baterai_bekas | 50 |
| 9 | botol_kaca_bening | 50 |
| 10 | botol_kaca_berwarna | 50 |
| 11 | botol_pembersih_lantai | 50 |
| 12 | botol_plastik_bening | 50 |
| 13 | botol_plastik_berwarna | 50 |
| 14 | botol_shampoo_sabun | 50 |
| 15 | botol_skincare_handbody | 50 |
| 16 | cangkang_telur | 50 |
| 17 | daun_kering_ranting | 50 |
| 18 | duri_kepala_ikan | 50 |
| 19 | gelas_plastik_minuman | 50 |
| 20 | jeriken_plastik | 50 |
| 21 | kain_perca_baju | 50 |
| 22 | kaleng_cat_kimia | 50 |
| 23 | kaleng_minuman_alumunium | 50 |
| 24 | kaleng_susu | 50 |
| 25 | kantong_plastik_hitam | 50 |
| 26 | kantong_plastik_warna | 50 |
| 27 | kardus_cokelat | 50 |
| 28 | karton_telur | 50 |
| 29 | kemasan_minuman_sachet | 50 |
| 30 | kemasan_minyak_goreng | 50 |
| 31 | kemasan_sachet_detergen | 50 |
| 32 | kemasan_snack | 50 |
| 33 | kemasan_tetra_pak | 50 |
| 34 | kertas_hvs_dokumen | 50 |
| 35 | koran_bekas | 50 |
| 36 | kulit_jeruk | 50 |
| 37 | kulit_mangga | 50 |
| 38 | lampu_bohlam_led | 50 |
| 39 | logam_rusak | 50 |
| 40 | masker_medis | 50 |
| 41 | paper_bag | 50 |
| 42 | puntung_rokok | 50 |
| 43 | sedotan_plastik | 50 |
| 44 | sikat_gigi_bekas | 50 |
| 45 | sisa_nasi | 50 |
| 46 | spons_pencuci_piring | 50 |
| 47 | tulang_ayam | 50 |
| 48 | tutup_botol_logam | 50 |
| 49 | wadah_kosmetik | 50 |
| 50 | wadah_plastik_rusak | 50 |

---

## 🛠️ Pembagian Dataset & Pipeline
Dataset dibagi menggunakan rasio akademis standar untuk validasi yang optimal:
- **Training Set (70%)**: 1.758 gambar (55 batch)
- **Validation Set (20%)**: 512 gambar (16 batch)
- **Test Set (10%)**: 241 gambar (8 batch)

### Konfigurasi Pipeline:
- **Batch Size**: 32
- **Image Size**: `(224, 224)`
- **Data Augmentation**:
  - Horizontal & Vertical Flip (`layers.RandomFlip`)
  - Random Rotation (`0.2`)
  - Random Zoom (`0.2`)
  - Random Translation (`0.1` horizontal & vertical)
- **Performance Optimization**: Prefetching data menggunakan `tf.data.AUTOTUNE` untuk mempercepat pemuatan batch pada GPU.

---

## 🧠 Arsitektur Model
Model dirancang menggunakan arsitektur modular yang menggabungkan model pra-latih (Transfer Learning) dan layer kustom:

```
Input Image (224x224x3)
        │
        ▼
Data Augmentation (Flip, Rotation, Zoom, Translation)
        │
        ▼
EfficientNetB0 Base (Pre-trained on ImageNet - Feature Extractor)
        │
        ▼
Global Average Pooling 2D
        │
        ▼
Batch Normalization
        │
        ▼
Dropout Layer (Rate: 0.4)
        │
        ▼
Dense Output Layer (50 Classes, Softmax)
```

### Ringkasan Jumlah Parameter:
- **Total parameter**: 4.118,741 (15,71 MB)
- **Trainable parameter (Phase 1)**: 66.610 (260,20 KB)
- **Non-trainable parameter**: 4.052.131 (15,46 MB)

---

## 📈 Alur Pelatihan (Training Workflow)

Pelatihan dilakukan dalam dua tahap utama dengan pemanfaatan callback pintar:

### 1. Tahap 1: Feature Extraction
- **Tujuan**: Melatih classifier head baru untuk menyesuaikan dataset sampah baru tanpa mengubah bobot dasar EfficientNet.
- **Optimizer**: Adam dengan Learning Rate **0.001**.
- **Epochs**: Maksimal 20 Epochs.
- **Callbacks**:
  - `ModelCheckpoint` untuk menyimpan model terbaik berdasarkan `val_loss` terendah (`best_efficientnet_model.h5`).
  - `EarlyStopping` dengan toleransi (`patience`) 5 epoch.
  - `ReduceLROnPlateau` dengan faktor `0.2` dan toleransi `3` epoch.
- **Hasil**: Berhenti pada Epoch 20, dengan performa terbaik dipulihkan dari **Epoch 19** (`val_loss: 0.1252` dan `val_accuracy: 96.09%`).

### 2. Tahap 2: Fine-Tuning
- **Tujuan**: Membuka blokir pembekuan (unfreeze) pada 20 layer teratas base model EfficientNetB0 untuk mengadaptasi filter visual agar lebih spesifik terhadap bentuk-bentuk sampah.
- **Optimizer**: Adam dengan Learning Rate sangat rendah **1e-5 (0.00001)** agar tidak merusak bobot yang telah matang.
- **Epochs**: Target tambahan 10 Epochs.
- **Hasil**: Proses dihentikan oleh `EarlyStopping` pada Epoch 25 karena kestabilan loss. Model terbaik yang dipulihkan berasal dari **Epoch 21** (`val_loss: 0.1443`).

---

## 🏆 Hasil Evaluasi Model

Model diuji menggunakan data **Test Set murni (241 gambar)** yang tidak pernah dilihat oleh model selama proses pelatihan untuk memastikan keandalan pengujian.

### Metrik Utama Evaluasi:
- **Akurasi Akhir (Accuracy)**: **97%**
- **Presisi Rata-rata (Macro Avg Precision)**: **94%**
- **Recall Rata-rata (Macro Avg Recall)**: **94%**
- **F1-Score Rata-rata (Macro Avg F1-Score)**: **93%**

### Analisis Klasifikasi Kunci:
- Mayoritas kelas (seperti `baterai_bekas`, `botol_kaca_bening`, `kaleng_susu`, dan `kulit_pisang`) mencapai nilai **F1-Score sempurna (1.00)**.
- Beberapa kelas dengan kemiripan visual tinggi (seperti `kemasan_minuman_sachet` vs `kemasan_snack` atau `struk_belanja` vs `kertas_hvs_dokumen`) mengalami sedikit misklasifikasi namun tetap berada dalam batas akurasi yang dapat ditoleransi (>75%).

---

## 🚀 Panduan Penggunaan & Uji Coba

Untuk menjalankan atau memodifikasi proyek ini secara lokal atau di Google Colab:

### 1. Prasyarat Sistem
Pastikan lingkungan python Anda telah terinstal library berikut:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn
```

### 2. Struktur Folder Dataset di Google Drive
Sebelum menjalankan kode, pastikan data gambar diletakkan dalam direktori Drive dengan format berikut:
```
Google Drive/
└── My Drive/
    └── Tugas/
        └── Dataset Sampah Rumah Tangga/
            ├── Styrofoam/
            ├── ampas_kopi_teh/
            ├── baterai_bekas/
            └── ... (50 folder kelas lainnya)
```

### 3. Cara Memprediksi Gambar Baru (Inference)
Di dalam notebook, panggil fungsi `prediksi_gambar_baru` untuk mengunggah gambar Anda sendiri secara interaktif:

```python
# Model akan otomatis disimpan setelah training di Drive:
# Path: /content/drive/MyDrive/Tugas/model_klasifikasi_sampah_terbaik.keras

# Load model dan jalankan fungsi prediksi
import tensorflow as tf
model_terload = tf.keras.models.load_model('/content/drive/MyDrive/Tugas/model_klasifikasi_sampah_terbaik.keras')

# Panggil fungsi prediksi
prediksi_gambar_baru(model_terload, class_names)
```
*Catatan: Fungsi tersebut akan memunculkan dialog unggah berkas, mengubah ukuran gambar ke 224x224, menampilkan visualisasi gambar dengan label prediksi, dan menunjukkan tingkat kepercayaan prediksi (Confidence Score).*

---

## 📂 Struktur Repositori
- `klasifikasi_sampah_rumah_tangga_dengan_Model_EfficientNetB0_menggunakan_Transfer_Learning_dan_Data_Augmentation.ipynb` : Berkas notebook utama berisi alur preprocessing, pelatihan, evaluasi, hingga deployment.
- `README.md` : Panduan dokumentasi proyek terlengkap (berkas ini).
