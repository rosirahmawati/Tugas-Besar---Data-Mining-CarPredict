import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image

st.set_page_config(
    page_title="India Price Predictor", 
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0f172a;
        color: #f1f5f9;
    }

    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 5px;
        text-align: left;
    }
    .main-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 25px;
        text-align: left;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: #1e293b;
        padding: 5px;
        border-radius: 8px;
        border-bottom: none;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 14px;
        border: none;
        padding: 0px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background-color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    .clean-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .prediction-card {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 24px;
        color: #f1f5f9;
        text-align: left;
    }
    
    div.stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        border-radius: 6px !important;
        padding: 10px 16px !important;
        transition: background-color 0.2s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2563eb !important;
    }
    
    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-weight: 500 !important;
        font-size: 13px !important;
        color: #cbd5e1 !important;
    }
    
    a {
        color: #3b82f6 !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    .text-justify {
        text-align: justify;
        text-justify: inter-word;
        line-height: 1.7;
        font-size: 14px;
        color: #cbd5e1;
        padding-left: 4px;
    }

    .section-body {
        padding-left: 2px;
        padding-right: 2px;
    }

    .profile-photo {
        border-radius: 10px;
        border: 2px solid #334155;
        object-fit: cover;
        display: block;
    }
    
    @media (max-width: 768px) {
        .block-container { padding: 1rem !important; }
        .main-title { font-size: 22px !important; }
        .main-subtitle { font-size: 12px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# 2. Artefak model
@st.cache_resource
def load_model():
    with open('model_car_predict.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_model()
scaler = artifacts['scaler']
kmeans = artifacts['kmeans']
models = artifacts['models']
fitur_regresi = artifacts['fitur_regresi']
fitur_cluster = artifacts['fitur_cluster']
metrics = artifacts.get('metrics', {})
eval_data = artifacts.get('eval_data', {})

# 3. Data mentah untuk EDA
@st.cache_data
def load_dataset():
    df_raw = pd.read_csv('car details v4.csv')
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    
    df_raw['Engine_Num'] = df_raw['Engine'].astype(str).str.replace(' cc', '', case=False)
    df_raw['Engine_Num'] = pd.to_numeric(df_raw['Engine_Num'].str.extract(r'([\d.]+)')[0], errors='coerce')
    
    df_raw['MaxPower_Num'] = df_raw['Max Power'].astype(str).str.replace(' bhp', '', case=False)
    df_raw['MaxPower_Num'] = pd.to_numeric(df_raw['MaxPower_Num'].str.extract(r'([\d.]+)')[0], errors='coerce')
    
    df_raw['Umur_Kendaraan'] = 2026 - df_raw['Year']
    
    for col in ['Price', 'Kilometer', 'Engine_Num', 'MaxPower_Num']:
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())
        
    for col in ['Fuel Type', 'Seller Type', 'Transmission', 'Owner']:
        df_raw[col] = df_raw[col].fillna(df_raw[col].mode()[0])
        
    # Klasifikasi kluster data mentah
    X_clust = df_raw[['Umur_Kendaraan', 'Kilometer', 'Engine_Num', 'MaxPower_Num']].copy()
    X_clust.columns = fitur_cluster
    X_clust_scaled = scaler.transform(X_clust)
    df_raw['Cluster'] = kmeans.predict(X_clust_scaled)
    
    return df_raw

df_data = load_dataset()

nama_kluster = {
    0: "Segmen Ekonomis (Mobil Lama / Jarak Tempuh Tinggi)",
    2: "Segmen Menengah (City Car / Mobil Keluarga)",
    1: "Segmen Premium (Mobil Mewah / Performa Tinggi)"
}

# 4. Header
st.markdown('<div class="main-title">Used Car Price Predictor (Indian Market - INR)</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Sistem Taksiran Harga Jual Wajar Mobil Bekas India menggunakan K-Means Clustering dan Regresi Linier dengan dataset CarDekho (2,059 Transaksi)</div>', unsafe_allow_html=True)

tab_predict, tab_eda, tab_model, tab_dev = st.tabs([
    "Kalkulator Estimasi", 
    "Analisis Data Pasar (EDA)", 
    "Evaluasi & Performa Model", 
    "Profile"
])

# Kalkulator Estimasi
with tab_predict:
    st.markdown("### Spesifikasi Kendaraan")
    st.markdown("Isi detail spesifikasi teknis mobil di bawah untuk menghitung taksiran harga jual wajar di pasar.")
    
    col_input, col_result = st.columns([1.2, 1])
    
    with col_input:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        
        # Grid input parameter
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            list_merk = sorted(list(df_data['Make'].unique()))
            merk = st.selectbox("Merk Kendaraan", list_merk, index=list_merk.index("Honda") if "Honda" in list_merk else 0)
        with r1_col2:
            year = st.slider("Tahun Perakitan", min_value=2000, max_value=2026, value=2018, step=1)
            
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            kilometer = st.number_input("Jarak Tempuh (KM)", min_value=0, max_value=1000000, value=45000, step=5000)
        with r2_col2:
            transmission = st.selectbox("Transmisi", ["Manual", "Automatic"])
            
        r3_col1, r3_col2 = st.columns(2)
        with r3_col1:
            engine = st.number_input("Kapasitas Mesin (CC)", min_value=600, max_value=6500, value=1200, step=100)
        with r3_col2:
            max_power = st.number_input("Tenaga Maksimum (BHP)", min_value=20, max_value=800, value=85, step=5)
            
        r4_col1, r4_col2, r4_col3 = st.columns(3)
        with r4_col1:
            fuel_type = st.selectbox("Bahan Bakar", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid", "CNG + CNG", "Petrol + CNG", "Petrol + LPG"])
        with r4_col2:
            seller_type = st.selectbox("Tipe Penjual", ["Individual", "Corporate", "Commercial Registration"])
        with r4_col3:
            owner = st.selectbox("Status Kepemilikan", ["First", "Second", "Third", "Fourth", "4 or More", "UnRegistered Car"])
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        hitung_button = st.button("Hitung Estimasi Harga", use_container_width=True)

    with col_result:
        if hitung_button:
            umur_kendaraan = 2026 - year
            
            merk_ada_di_data = merk in df_data['Make'].values
            
            # Batasan wajar spesifikasi pasar berdasarkan dataset
            engine_min_data = int(df_data['Engine_Num'].quantile(0.01))
            engine_max_data = int(df_data['Engine_Num'].quantile(0.99))
            power_min_data = int(df_data['MaxPower_Num'].quantile(0.01))
            power_max_data = int(df_data['MaxPower_Num'].quantile(0.99))
            km_max_data = int(df_data['Kilometer'].quantile(0.99))
            
            warnings_list = []
            if not merk_ada_di_data:
                warnings_list.append(f"Merk '{merk}' tidak terdapat dalam dataset pelatihan (CarDekho India). Hasil estimasi mungkin kurang akurat.")
            if engine < engine_min_data or engine > engine_max_data:
                warnings_list.append(f"Kapasitas mesin {engine} CC di luar rentang data pasar India ({engine_min_data}\u2013{engine_max_data} CC). Estimasi tidak dapat dijamin akurat.")
            if max_power < power_min_data or max_power > power_max_data:
                warnings_list.append(f"Tenaga mesin {max_power} BHP di luar rentang data pasar India ({power_min_data}\u2013{power_max_data} BHP). Estimasi tidak dapat dijamin akurat.")
            if kilometer > km_max_data:
                warnings_list.append(f"Jarak tempuh {kilometer:,} KM melampaui batas atas data pelatihan ({km_max_data:,} KM).")
            if year < 2005:
                warnings_list.append("Tahun perakitan sangat lama (< 2005). Data mobil sangat tua sangat sedikit dalam dataset.")
            
            # Tampilkan peringatan jika ada
            if warnings_list:
                for w in warnings_list:
                    st.warning(f"\u26a0\ufe0f {w}", icon=None)
            
            # 1. Klasifikasikan segmen pasar kendaraan (K-Means)
            input_cluster = pd.DataFrame([[umur_kendaraan, kilometer, engine, max_power]], columns=fitur_cluster)
            input_cluster_scaled = scaler.transform(input_cluster)
            cluster_terpilih = kmeans.predict(input_cluster_scaled)[0]
            
            # 2. Data regresi dengan dummy variables
            input_regresi = pd.DataFrame(0, index=[0], columns=fitur_regresi)
            
            if 'Umur_Kendaraan' in input_regresi.columns: input_regresi['Umur_Kendaraan'] = umur_kendaraan
            if 'Kilometer' in input_regresi.columns: input_regresi['Kilometer'] = kilometer
            if 'Engine' in input_regresi.columns: input_regresi['Engine'] = engine
            if 'Max Power' in input_regresi.columns: input_regresi['Max Power'] = max_power
            
            kolom_fuel = f"Fuel Type_{fuel_type}"
            if kolom_fuel in input_regresi.columns: input_regresi[kolom_fuel] = 1
                
            kolom_seller = f"Seller Type_{seller_type}"
            if kolom_seller in input_regresi.columns: input_regresi[kolom_seller] = 1
                
            kolom_trans = f"Transmission_{transmission}"
            if kolom_trans in input_regresi.columns: input_regresi[kolom_trans] = 1
                
            kolom_owner = f"Owner_{owner}"
            if kolom_owner in input_regresi.columns: input_regresi[kolom_owner] = 1
            
            # 3. Prediksi harga menggunakan model regresi linier spesifik kluster
            model_regresi = models[cluster_terpilih]
            prediksi_harga = model_regresi.predict(input_regresi)[0]

            if prediksi_harga < 0:
                st.error("Hasil kalkulasi menghasilkan nilai negatif. Kombinasi spesifikasi yang dimasukkan (usia, jarak tempuh, atau tenaga mesin) kemungkinan berada jauh di luar pola data pelatihan sehingga model tidak dapat menghasilkan estimasi yang valid.")
                prediksi_harga = None
            
            if prediksi_harga is not None:
                # Konversi taksiran rupiah (1 INR = Rp 188.5)
                rupiah_est = prediksi_harga * 188.5
                
                # Format Rupee India (Lakh / Crore)
                if prediksi_harga >= 10000000:
                    lakh_est = f"\u20b9 {prediksi_harga/10000000:.2f} Crore"
                else:
                    lakh_est = f"\u20b9 {prediksi_harga/100000:.2f} Lakh"
                    
                badge_class = f"badge-{cluster_terpilih}"

                st.markdown(f"""
                    <div class="prediction-card">
                        <span class="cluster-badge {badge_class}">{nama_kluster[cluster_terpilih].split(" (")[0]}</span>
                        <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Hasil Estimasi Jual Wajar</div>
                        <div style="font-size: 32px; font-weight: 700; color: #10b981; margin: 4px 0;">{lakh_est}</div>
                        <div style="font-size: 14px; color: #cbd5e1;">Detail Nominal: INR {prediksi_harga:,.2f}</div>
                        <div style="font-size: 14px; color: #60a5fa; margin-top: 2px;">Estimasi Rupiah: Rp {rupiah_est:,.0f} (Kurs Rp188.5)</div>
                        <hr style="border-color: #334155; margin: 16px 0;">
                        <div style="font-size: 11px; color: #64748b; line-height: 1.4;">
                            Estimasi ini dihasilkan menggunakan model regresi linier kluster berdasarkan analisis statistik pasar otomotif India. Nilai hanya berlaku untuk kendaraan dengan spesifikasi sebanding data pasar India.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Perbandingan visual
                df_cluster_subset = df_data[df_data['Cluster'] == cluster_terpilih]
                harga_rata_cluster = df_cluster_subset['Price'].mean()
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=['Taksiran Kendaraan', 'Rata-rata Segmen'],
                    y=[prediksi_harga, harga_rata_cluster],
                    marker_color=['#10b981', '#3b82f6'],
                    text=[f"INR {prediksi_harga:,.0f}", f"INR {harga_rata_cluster:,.0f}"],
                    textposition='inside',
                    width=0.4
                ))
                fig_bar.update_layout(
                    title=dict(
                        text="Harga Taksiran vs. Rata-rata Segmen Pasar",
                        font=dict(size=14, family='Inter', color='#f1f5f9')
                    ),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=220,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
        else:
            st.markdown("""
                <div style="border: 1px dashed #334155; border-radius: 8px; padding: 40px; text-align: center; height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="font-size: 14px; color: #94a3b8; font-weight: 500;">Menunggu Input Spesifikasi Kendaraan</div>
                    <div style="color: #64748b; font-size: 12px; margin-top: 6px; max-width: 300px;">
                        Tentukan parameter kendaraan pada panel sebelah kiri dan klik tombol "Hitung Estimasi Harga" untuk memulai kalkulasi.
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Analisis Data Pasar (EDA)
with tab_eda:
    st.markdown("### Analisis Eksploratif Data Pasar")
    st.markdown("Pemetaan pola harga jual mobil bekas berdasarkan 2,059 transaksi pasar otomotif sekunder.")
    
    # Metrik Ringkasan
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric(label="Jumlah Sampel Transaksi", value=f"{len(df_data):,}")
    with col_s2:
        st.metric(label="Rata-rata Harga Pasar", value=f"₹ {df_data['Price'].mean()/100000:.2f} Lakh")
    with col_s3:
        st.metric(label="Jumlah Pabrikan (Make)", value=f"{df_data['Make'].nunique()}")
    with col_s4:
        st.metric(label="Rata-rata Usia Kendaraan", value=f"{df_data['Umur_Kendaraan'].mean():.1f} Tahun")
        
    st.write("---")
    
    c1_col1, c1_col2 = st.columns(2)
    
    with c1_col1:
        # Sebaran harga per segmen
        df_data['Nama_Cluster'] = df_data['Cluster'].map(nama_kluster)
        fig_box = px.box(
            df_data, 
            x='Nama_Cluster', 
            y='Price',
            color='Nama_Cluster',
            title='Sebaran Harga Jual Berdasarkan Segmen Pasar',
            labels={'Price': 'Harga Jual (INR)', 'Nama_Cluster': 'Segmen Pasar'},
            color_discrete_map={
                nama_kluster[0]: '#ef4444',
                nama_kluster[2]: '#f59e0b',
                nama_kluster[1]: '#10b981'
            }
        )
        fig_box.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with c1_col2:
        # Usia vs Harga
        fig_scatter_age = px.scatter(
            df_data, 
            x='Umur_Kendaraan', 
            y='Price', 
            color='Nama_Cluster',
            trendline='ols',
            title='Hubungan Usia Kendaraan vs Harga Jual (Tren Depresiasi)',
            labels={'Umur_Kendaraan': 'Usia Kendaraan (Tahun)', 'Price': 'Harga Jual (INR)'},
            opacity=0.6,
            color_discrete_map={
                nama_kluster[0]: '#ef4444',
                nama_kluster[2]: '#f59e0b',
                nama_kluster[1]: '#10b981'
            }
        )
        fig_scatter_age.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter_age, use_container_width=True)
        
    c2_col1, c2_col2 = st.columns(2)
    
    with c2_col1:
        # Tenaga vs Kilometer
        fig_scatter_power = px.scatter(
            df_data,
            x='MaxPower_Num',
            y='Kilometer',
            color='Nama_Cluster',
            title='Distribusi Tenaga Maksimum (BHP) vs Jarak Tempuh (KM)',
            labels={'MaxPower_Num': 'Tenaga Kendaraan (BHP)', 'Kilometer': 'Jarak Tempuh (KM)'},
            opacity=0.6,
            color_discrete_map={
                nama_kluster[0]: '#ef4444',
                nama_kluster[2]: '#f59e0b',
                nama_kluster[1]: '#10b981'
            }
        )
        fig_scatter_power.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter_power, use_container_width=True)
        
    with c2_col2:
        # Korelasi
        kolom_korelasi = ['Price', 'Umur_Kendaraan', 'Kilometer', 'Engine_Num', 'MaxPower_Num']
        df_corr = df_data[kolom_korelasi].corr()
        fig_heatmap = px.imshow(
            df_corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Korelasi Pearson Antar Atribut Numerik',
            labels=dict(color="Koefisien")
        )
        fig_heatmap.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Evaluasi & Performa Model
with tab_model:
    st.markdown("### Evaluasi Model Prediksi")
    st.markdown("Arsitektur model menggunakan kombinasi **K-Means Clustering** untuk pembagian kelompok pasar wajar, dilanjutkan dengan model **Regresi Linier** terpisah pada masing-masing kluster.")
    
    st.markdown("#### Ringkasan Metrik Evaluasi Data Uji (Test Set 20%)")
    
    # Pembuatan tabel metrik formal
    tabel_metrik = []
    for cid, m in metrics.items():
        tabel_metrik.append({
            "Kluster": f"Kluster {cid}",
            "Kategori Segmen": nama_kluster[cid].split(" (")[0],
            "R-Squared (R²)": f"{m['r2']:.4f}",
            "Mean Absolute Error (MAE)": f"INR {m['mae']:,.2f}",
            "Root Mean Squared Error (RMSE)": f"INR {m['rmse']:,.2f}"
        })
    st.table(pd.DataFrame(tabel_metrik))
    
    st.write("---")
    
    st.markdown("#### Plot Sebaran Aktual vs Prediksi")
    cluster_view = st.selectbox("Pilih Kluster:", [0, 2, 1], format_func=lambda x: nama_kluster[x].split(" (")[0])
    
    if cluster_view in eval_data:
        y_test_vals = eval_data[cluster_view]['y_test']
        y_pred_vals = eval_data[cluster_view]['y_pred']
        
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=y_test_vals,
            y=y_pred_vals,
            mode='markers',
            marker=dict(color='rgba(59, 130, 246, 0.5)', size=7),
            name='Nilai Prediksi'
        ))
        
        min_val = min(min(y_test_vals), min(y_pred_vals))
        max_val = max(max(y_test_vals), max(y_pred_vals))
        fig_eval.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#ef4444', width=1.5, dash='dash'),
            name='Garis Ideal (Y = X)'
        ))
        
        fig_eval.update_layout(
            title=f"Actual vs. Predicted Plot - {nama_kluster[cluster_view].split(' (')[0]}",
            xaxis_title="Harga Aktual (INR)",
            yaxis_title="Harga Prediksi (INR)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_eval, use_container_width=True)

# Profile Pengembang
with tab_dev:
    st.markdown("### Informasi Proyek & Pengembang")

    st.markdown('<div class="section-body">', unsafe_allow_html=True)
    st.markdown("""<div class="text-justify">
    Proyek ini adalah <strong>Sistem Prediksi Harga Jual Wajar Mobil Bekas</strong> berbasis Machine Learning 
    yang dikembangkan sebagai Tugas Besar mata kuliah Penambangan Data (Kelas BBK2LAB3), Universitas Telkom.
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="text-justify" style="margin-top: 12px;">
    <strong>Permasalahan yang diangkat:</strong> Pada platform jual-beli mobil bekas seperti OLX Autos atau Carmudi, 
    harga yang dipasang penjual sangat subjektif &mdash; ditentukan sepihak berdasarkan perkiraan pribadi, bukan data 
    pasar. Kondisi ini menciptakan asimetri informasi yang merugikan pembeli yang tidak memiliki acuan harga terverifikasi.
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="text-justify" style="margin-top: 12px;">
    <strong>Solusi yang dibangun:</strong> Aplikasi ini mengimplementasikan pendekatan <strong>Hybrid Machine Learning</strong> 
    dua tahap: pertama, <strong>K-Means Clustering</strong> mengelompokkan kendaraan ke dalam 3 segmen pasar 
    (Ekonomis, Menengah, Premium) berdasarkan spesifikasi fisik. Kemudian, <strong>Regresi Linier per Kluster</strong> 
    melatih model terpisah pada setiap segmen untuk menghasilkan taksiran harga yang lebih akurat dibandingkan model tunggal. 
    Dataset yang digunakan adalah <code>car details v4.csv</code> dari CarDekho (Kaggle), 2,059 catatan transaksi 
    mobil bekas di India dalam mata uang Indian Rupee (INR).
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")

    st.markdown("#### Detail Akademik & Pengembang")
    
    photo_col, info_col = st.columns([1, 3])
    
    with photo_col:
        photo_path = "Photo.jpg"
        if os.path.exists(photo_path):
            img = Image.open(photo_path)
            st.image(img, width=130)
        else:
            st.markdown("<div style='font-size: 50px; text-align:center;'>&#128105;&#8205;&#128187;</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='margin-top: 8px;'>
            <div style='font-weight: 600; font-size: 14px; color: #f1f5f9;'>By: Rosi Rahmawati</div>
            <div style='margin-top: 2px;'><a href="https://www.linkedin.com/in/rosi-rahmawati/" target="_blank" style='color: #3b82f6; font-size: 12px; font-weight: 500; text-decoration: none;'>🔗 LinkedIn Profile</a></div>
            <div style='margin-top: 2px;'><a href="https://github.com/rosirahmawati/Tugas-Besar---Data-Mining-CarPredict.git" target="_blank" style='color: #3b82f6; font-size: 12px; font-weight: 500; text-decoration: none;'>🐱 GitHub Repository</a></div>
            </div>
        """, unsafe_allow_html=True)
    
    with info_col:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Akademik**")
            st.markdown("""
            - Institusi: Telkom University
            - Prodi: S1 Sistem Informasi
            - Mata Kuliah: Penambangan Data (BBK2LAB3)
            - Metodologi: CRISP-DM
            - Tahun: 2026
            """)
        with col_b:
            st.markdown("**Tech Stack**")
            st.markdown("""
            - Python (Pandas, NumPy)
            - Scikit-Learn (KMeans, LinearRegression)
            - Streamlit + Plotly
            - StandardScaler, One-Hot Encoding
            - Pickle (Model Serialization)
            """)
    
