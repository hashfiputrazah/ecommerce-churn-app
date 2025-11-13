
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. Load Model
# Model yang di-load adalah pipeline lengkap (preprocessor + model CatBoost Tuned)
model_path = 'churn_prediction_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Error: Model '{model_path}' tidak ditemukan. Pastikan sudah di-pickle.")
    st.stop()

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 2. Definisikan Opsi Kategorikal (Sesuai dengan data asli)
# Data kategorikal yang dibutuhkan model (PreferedOrderCat, MaritalStatus)
cat_options = {
    'PreferedOrderCat': ['Mobile Phone', 'Laptop & Accessory', 'Fashion', 'Grocery', 'Others'],
    'MaritalStatus': ['Single', 'Married', 'Divorced']
}

# 3. Streamlit Interface
st.set_page_config(page_title="Prediksi Customer Churn E-Commerce", layout="wide")
st.title("🛒 Prediksi Customer Churn E-Commerce")
st.markdown("---")
st.subheader("Masukkan data pelanggan untuk memprediksi probabilitas Churn.")

# 4. Input Form Layout
with st.form("churn_form"):
    st.header("Detail Data Pelanggan")
    col1, col2, col3 = st.columns(3)
    
    # Input Kolom Numerik
    with col1:
        Tenure = st.slider("1. Tenure (Bulan)", 0, 70, 15, key='t1')
        WarehouseToHome = st.slider("2. Jarak Gudang ke Rumah", 5, 40, 20, key='t2')
        NumberOfDeviceRegistered = st.number_input("3. Jumlah Perangkat Terdaftar", 1, 10, 4, step=1, key='t3')
    
    with col2:
        SatisfactionScore = st.slider("4. Skor Kepuasan (1=Tidak Puas, 5=Puas)", 1, 5, 3, step=1, key='t4')
        NumberOfAddress = st.number_input("5. Jumlah Alamat Terdaftar", 1, 15, 3, step=1, key='t5')
        # Complain adalah numerik (0/1)
        Complain = st.selectbox("6. Ada Keluhan (Bulan Terakhir)?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", key='t6')
        
    with col3:
        DaySinceLastOrder = st.slider("7. Hari Sejak Order Terakhir", 0.0, 30.0, 7.0, step=0.5, key='t7')
        CashbackAmount = st.slider("8. Jumlah Cashback (Rupiah)", 100.0, 350.0, 180.0, step=0.1, key='t8')
        
        # Input Kolom Kategorikal
        PreferedOrderCat = st.selectbox("9. Kategori Pesanan Favorit", cat_options['PreferedOrderCat'], key='t9')
        MaritalStatus = st.selectbox("10. Status Pernikahan", cat_options['MaritalStatus'], key='t10')
    
    submitted = st.form_submit_button("Prediksi Churn")

# 5. Prediction Logic
if submitted:
    # Buat DataFrame input (Pastikan nama dan urutan kolom sesuai dengan X_train)
    data_input = {
        'Tenure': Tenure,
        'WarehouseToHome': WarehouseToHome,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'SatisfactionScore': SatisfactionScore,
        'NumberOfAddress': NumberOfAddress,
        'Complain': Complain,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount,
        'PreferedOrderCat': PreferedOrderCat,
        'MaritalStatus': MaritalStatus
    }
    # Buat DataFrame dengan urutan kolom yang benar (sesuai X_train)
    # Urutan kolom ini PENTING karena pipeline di dalam model akan memprosesnya
    column_order = [
        'Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered', 
        'SatisfactionScore', 'NumberOfAddress', 'Complain', 
        'DaySinceLastOrder', 'CashbackAmount', 
        'PreferedOrderCat', 'MaritalStatus'
    ]
    input_df = pd.DataFrame([data_input], columns=column_order)

    # Prediksi
    try:
        # Prediksi probabilitas kelas 1 (Churn)
        proba = model.predict_proba(input_df)[:, 1][0]
        # Prediksi kelas (0 atau 1)
        prediction = model.predict(input_df)[0] 

        # Tampilkan Hasil
        st.markdown("---")
        st.subheader("Hasil Prediksi")
        
        # Interpretasi
        if prediction == 1:
            st.error(f"⚠️ **PELANGGAN DIPREDIKSI AKAN CHURN!**")
            st.metric(label="Probabilitas Churn", value=f"{proba*100:.2f}%")
            st.markdown("**Rekomendasi:** Segera lakukan intervensi retensi (diskon, penawaran khusus) untuk pelanggan ini. Prioritaskan berdasarkan nilai `Tenure` yang rendah dan adanya `Complain`.")
        else:
            st.success(f"✅ **PELANGGAN DIPREDIKSI TIDAK AKAN CHURN**")
            st.metric(label="Probabilitas Churn", value=f"{proba*100:.2f}%")
            st.markdown("**Rekomendasi:** Pelanggan ini relatif loyal. Terus pantau dan pastikan kualitas layanan tetap tinggi.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.exception(e)

