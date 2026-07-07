import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load data mobil bekas dari file CSV
df = pd.read_csv('car details v4.csv')

# 2. Bersihin data dan prapemrosesan (data cleaning & preprocessing)
# Buang data yang duplikat biar gak bias
df = df.drop_duplicates().reset_index(drop=True)

# Bersihin kolom Engine (buang tulisan ' cc' dan ubah jadi angka)
df['Engine'] = df['Engine'].astype(str).str.replace(' cc', '', case=False)
df['Engine'] = pd.to_numeric(df['Engine'].str.extract(r'([\d.]+)')[0], errors='coerce')

# Bersihin kolom Max Power (buang tulisan ' bhp' dan ubah jadi angka)
df['Max Power'] = df['Max Power'].astype(str).str.replace(' bhp', '', case=False)
df['Max Power'] = pd.to_numeric(df['Max Power'].str.extract(r'([\d.]+)')[0], errors='coerce')

# Isi data yang kosong (NaN) pake nilai median (untuk numerik) dan mode (untuk kategori)
for col in ['Price', 'Kilometer', 'Engine', 'Max Power']:
    df[col] = df[col].fillna(df[col].median())

for col in ['Fuel Type', 'Seller Type', 'Transmission', 'Owner']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Hitung umur kendaraan (kita pake basis tahun 2026 sesuai waktu sekarang)
df['Umur_Kendaraan'] = 2026 - df['Year']

# Mengatasi outlier di kolom Price dan Kilometer pake metode clipping (persentil 1% dan 99%)
for col in ['Price', 'Kilometer']:
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    df[col] = np.where(df[col] < q_low, q_low, df[col])
    df[col] = np.where(df[col] > q_high, q_high, df[col])

# Ambil kolom yang mau kita pake buat model
kolom_dipertahankan = ['Price', 'Umur_Kendaraan', 'Kilometer', 'Engine', 'Max Power', 
                        'Fuel Type', 'Seller Type', 'Transmission', 'Owner']
df_fokus = df[kolom_dipertahankan].copy()

# Ubah kolom kategori jadi dummy variabel (one-hot encoding)
df_encoded = pd.get_dummies(df_fokus, columns=['Fuel Type', 'Seller Type', 'Transmission', 'Owner'], drop_first=True)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# 3. Pengelompokan mobil pake K-Means Clustering
# Kita kelompokkan mobil berdasarkan umur, kilometer, kapasitas mesin, dan tenaganya
fitur_cluster = ['Umur_Kendaraan', 'Kilometer', 'Engine', 'Max Power']
X_cluster = df_encoded[fitur_cluster]

# Standarisasi fitur cluster biar skalanya sama
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Latih K-Means dengan 3 kluster (Ekonomis, Menengah, Premium)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, algorithm='elkan', random_state=42)
df_encoded['Cluster_Label'] = kmeans.fit_predict(X_cluster_scaled)

# 4. Pelatihan Regresi Linear dan Evaluasi per Kluster
# Fitur regresi adalah semua kolom kecuali target (Price) dan Cluster_Label
fitur_regresi = [col for col in df_encoded.columns if col not in ['Price', 'Cluster_Label']]

models = {}
metrics = {}
eval_data = {}

for cluster_id in sorted(df_encoded['Cluster_Label'].unique()):
    df_sub = df_encoded[df_encoded['Cluster_Label'] == cluster_id]
    X_r = df_sub[fitur_regresi]
    y_r = df_sub['Price']
    
    # Bagi data jadi 80% train dan 20% test buat evaluasi performa model
    X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    
    # Latih model regresi linear sementara untuk evaluasi
    eval_model = LinearRegression()
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    
    # Hitung metrik evaluasi (R-Squared, MAE, RMSE)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Simpan metrik hasil evaluasi
    metrics[cluster_id] = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse)
    }
    
    # Simpan data aktual vs prediksi buat digambar di chart app.py nanti
    eval_data[cluster_id] = {
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }
    
    # Latih model regresi linear akhir pake seluruh data kluster ini biar hasilnya maksimal buat deploy
    final_model = LinearRegression()
    final_model.fit(X_r, y_r)
    models[cluster_id] = final_model

# 5. Simpan semua model dan data pendukung ke dalam file pickle
artifacts = {
    'scaler': scaler,
    'kmeans': kmeans,
    'models': models,
    'fitur_regresi': fitur_regresi,
    'fitur_cluster': fitur_cluster,
    'metrics': metrics,
    'eval_data': eval_data,
    'raw_data_summary': {
        'total_rows': len(df),
        'cluster_counts': df_encoded['Cluster_Label'].value_counts().to_dict()
    }
}

with open('model_car_predict.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("BERHASIL! Model berhasil dilatih dan dievaluasi per kluster.")
for cid, m in metrics.items():
    print(f"Kluster {cid}: R2={m['r2']:.4f}, MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}")