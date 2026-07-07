# Used Car Market Segmentation & Price Intelligence Engine 

Tugas Besar Mata Kuliah Penambangan Data (Data Mining)
Prodi S1 Sistem Informasi, Universitas Telkom, Bandung (2026)
Dibuat Oleh : Rosi Rahmawati

---

### Ringkasan Proyek

Penentuan harga mobil bekas pada e-commerce otomotif sering kali bersifat subjektif, didasarkan pada perkiraan pribadi penjual. Hal ini memicu asimetri informasi yang merugikan baik bagi penjual maupun pembeli.

Aplikasi ini adalah **Sistem Taksiran Harga Mobil Bekas** interaktif yang dibangun menggunakan pendekatan **Hybrid Machine Learning**:

1. **K-Means Clustering**: Segmentasi pasar otomatis berdasarkan aspek fisik (Usia Kendaraan, Jarak Tempuh, Kapasitas Mesin, dan Tenaga Maksimum) ke dalam 3 kelas pasar (Ekonomis, Menengah, dan Premium).
2. **Linear Regression**: Prediksi nominal harga jual wajar menggunakan model regresi linier spesifik yang dilatih terpisah pada masing-masing kluster segmen pasar.

Proyek ini mengikuti metodologi standar industri **CRISP-DM (Cross-Industry Standard Process for Data Mining)**.

---

### Metodologi CRISP-DM

1. **Business Understanding (Analisis Kebutuhan)**: Mengatasi ketidakpastian harga mobil bekas dengan menyediakan sistem estimasi harga objektif berbasis spesifikasi kendaraan, guna meminimalisir bias subjektif penjual.
2. **Data Understanding (Pemahaman Data)**: Menggunakan dataset `car details v4.csv` dari CarDekho (Kaggle) berisi data historis transaksi mobil bekas di India dengan kolom harga (target), usia, jarak tempuh, tipe transmisi, bahan bakar, daya mesin, dll.
3. **Data Preparation (Persiapan Data)**:
   - Penghapusan baris data duplikat.
   - Ekstrak nilai numerik dari string (kolom `Engine` cc dan `Max Power` bhp).
   - Pengisian nilai kosong (_missing values_) menggunakan median (numerik) dan modus (kategori).
   - Penanganan outlier pada kolom harga dan kilometer menggunakan metode _clipping_ persentil 1% dan 99%.
   - Transformasi variabel kategori menggunakan _One-Hot Encoding_.
4. **Modeling (Pemodelan)**:
   - Pengelompokan data ke dalam 3 kluster menggunakan _K-Means Clustering_ dengan penskalaan _StandardScaler_.
   - Pelatihan model _Linear Regression_ untuk setiap kluster secara terpisah menggunakan library `scikit-learn`.
5. **Evaluation (Evaluasi)**: Menguji tingkat akurasi model pada porsi _test set_ (20%) menggunakan metrik R-Squared ($R^2$), Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE).
6. **Deployment (Penerapan)**: Implementasi model ke dalam dashboard web interaktif menggunakan framework **Streamlit** dengan visualisasi interaktif dari **Plotly**.

---

### Fitur Streamlit

- **Prediksi Harga (Price Predictor)**: Input detail spesifikasi mobil bekas Anda untuk menghasilkan taksiran harga secara instan, dilengkapi perbandingan visual dengan harga rata-rata kelas pasarnya. Prediksi disajikan dalam mata uang Rupee India (INR/Lakhs) serta estimasi Rupiah (IDR).
- **Analisis Pasar (Market Insights)**: Visualisasi data interaktif (EDA) seperti sebaran harga tiap segmen pasar, tren depresiasi harga berdasarkan usia kendaraan, dan matriks korelasi antar variabel.
- **Performa Model**: Laporan transparansi metrik evaluasi model ($R^2$, MAE, RMSE) untuk masing-masing kluster segmen pasar beserta plot sebaran aktual vs prediksi.
- **Profil Developer**: Halaman informasi pengembang proyek mahasiswa Universitas Telkom lengkap dengan bio, NIM, dan tech stack.

---

## Cara Menjalankan Aplikasi Secara Lokal / How to Run Locally

### 1. Prasyarat / Prerequisites

Pastikan Anda sudah menginstal Python 3.8+ di komputer Anda.  
_Ensure Python 3.8+ is installed on your machine._

### 2. Kloning Repositori / Clone Repository

```bash
git clone https://github.com/rosirahmawati/Used-Car-Price-Prediction.git
cd Used-Car-Price-Prediction
```

### 3. Instal Dependensi / Install Dependencies

Instal semua pustaka yang dibutuhkan menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Pelatihan Model / Model Training

Sebelum menjalankan web app, latih model terlebih dahulu untuk menghasilkan file artefak `model_car_predict.pkl`:

```bash
python train.py
```

### 5. Jalankan Aplikasi / Launch Streamlit App

Jalankan server lokal Streamlit:

```bash
streamlit run app.py
```

Aplikasi secara otomatis akan terbuka di browser pada alamat `http://localhost:8501`.

---
