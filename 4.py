# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Kualitas Air Kolam Ikan",
    page_icon="💧",
    layout="wide"
)

# --- FUNSI PEMBANTU (BERSAMA UNTUK SEMUA HALAMAN) ---

@st.cache_data
def load_full_data():
    """Memuat dan membersihkan seluruh dataset."""
    try:
        df = pd.read_excel('WQD.xlsx')
        df.columns = df.columns.str.strip()
        
        quality_map = {'Excellent': 0, 'Good': 1, 'Poor': 2}
        reverse_map = {v: k for k, v in quality_map.items()}

        if df['Water Quality'].dtype == 'object':
            df['Water Quality'] = df['Water Quality'].map(quality_map)

        df['Quality Label'] = df['Water Quality'].map(reverse_map)
        
        return df
    except FileNotFoundError:
        st.error("File data 'WQD.xlsx' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None

@st.cache_resource
def load_model():
    """Memuat model yang sudah dilatih."""
    try:
        model = joblib.load('water_quality_model.pkl')
        return model
    except FileNotFoundError:
        st.error("File model 'water_quality_model.pkl' tidak ditemukan. Silakan jalankan 'train_model.py' terlebih dahulu.")
        return None

@st.cache_data
def get_feature_names():
    """Mendapatkan nama fitur dari file Excel."""
    try:
        df = pd.read_excel('WQD.xlsx')
        df.columns = df.columns.str.strip()
        return df.drop('Water Quality', axis=1).columns.tolist()
    except FileNotFoundError:
        st.error("File data 'WQD.xlsx' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None

@st.cache_data
def get_sample_data(_feature_names):
    """Mengambil satu sampel untuk setiap kategori kualitas dengan cara yang lebih aman."""
    try:
        df = load_full_data()
        if df is None or _feature_names is None:
            return None
            
        # Memilih hanya kolom fitur yang diperlukan, lebih aman dari drop()
        sample_excellent = df[df['Water Quality'] == 0].iloc[0:1][_feature_names]
        sample_good = df[df['Water Quality'] == 1].iloc[0:1][_feature_names]
        sample_poor = df[df['Water Quality'] == 2].iloc[0:1][_feature_names]
        
        return {
            "Excellent": sample_excellent,
            "Good": sample_good,
            "Poor": sample_poor
        }
    except Exception as e:
        st.error(f"Gagal memuat contoh data: {e}")
        return None

# --- MUAT DATA DAN MODEL DI AWAL ---
full_data = load_full_data()
model = load_model()
feature_names = get_feature_names()
# Pass feature_names ke fungsi untuk memastikan cache diperbarui dengan benar
sample_data_dict = get_sample_data(_feature_names=feature_names)

# --- SIDEBAR UNTUK NAVIGASI ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Prediksi Kualitas Air", "Dashboard & Data"])

# --- HALAMAN 1: PREDIKSI KUALITAS AIR ---
if page == "Prediksi Kualitas Air":
    st.title("💧 Sistem Prediksi Kualitas Air Kolam Ikan")
    st.markdown("""Aplikasi ini menggunakan model Machine Learning untuk memprediksi kualitas air di kolam ikan berdasarkan 14 parameter fisika dan kimia.""")

    if model and feature_names and sample_data_dict:
        st.sidebar.header("Input Parameter Kualitas Air")
        input_mode = st.sidebar.radio("Pilih Mode Input:", ("Input Manual", "Contoh Data (Excellent)", "Contoh Data (Good)", "Contoh Data (Poor)"))

        def user_input_features(mode):
            if mode == "Input Manual":
                temp = st.sidebar.slider('Suhu Air (Temp)', 0.0, 40.0, 25.0)
                turbidity = st.sidebar.slider('Kekeruhan (Turbidity cm)', 0.0, 100.0, 30.0)
                do = st.sidebar.slider('Oksigen Terlarut (DO mg/L)', 0.0, 20.0, 5.0)
                bod = st.sidebar.slider('Kebutuhan Oksigen Biokimia (BOD mg/L)', 0.0, 20.0, 4.0)
                co2 = st.sidebar.slider('Karbon Dioksida (CO2)', 0.0, 50.0, 10.0)
                ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0)
                alkalinity = st.sidebar.slider('Alkalinitas (mg/L)', 0.0, 300.0, 100.0)
                hardness = st.sidebar.slider('Kesadahan (Hardness mg/L)', 0.0, 300.0, 150.0)
                calcium = st.sidebar.slider('Kalsium (Calcium mg/L)', 0.0, 200.0, 50.0)
                ammonia = st.sidebar.slider('Amonia (Ammonia mg/L)', 0.0, 5.0, 0.5)
                nitrite = st.sidebar.slider('Nitrit (Nitrite mg/L)', 0.0, 2.0, 0.2)
                phosphorus = st.sidebar.slider('Fosfor (Phosphorus mg/L)', 0.0, 5.0, 0.5)
                h2s = st.sidebar.slider('Hidrogen Sulfida (H2S mg/L)', 0.0, 1.0, 0.1)
                plankton = st.sidebar.slider('Plankton (No./L)', 0, 2000, 500)
                input_values = [temp, turbidity, do, bod, co2, ph, alkalinity, hardness, calcium, ammonia, nitrite, phosphorus, h2s, plankton]
                features = pd.DataFrame([input_values], columns=feature_names)
                return features
            else:
                sample_key = mode.split("(")[1].split(")")[0]
                if sample_data_dict and sample_key in sample_data_dict:
                    st.sidebar.info(f"Menampilkan contoh data untuk kualitas '{sample_key}'.")
                    return sample_data_dict[sample_key]
                return pd.DataFrame()

        input_df = user_input_features(input_mode)

        with st.expander("Lihat Parameter yang Anda Masukkan"):
            st.write(input_df)

        if st.button('Prediksi Kualitas Air'):
            if not input_df.empty:
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                quality_labels = {0: 'Excellent', 1: 'Good', 2: 'Poor'}
                predicted_quality = quality_labels[prediction[0]]

                st.subheader('Hasil Prediksi Kualitas Air')
                if predicted_quality == 'Excellent':
                    st.success(f"Kualitas Air: **{predicted_quality}** ✨")
                elif predicted_quality == 'Good':
                    st.info(f"Kualitas Air: **{predicted_quality}** 👍")
                else:
                    st.error(f"Kualitas Air: **{predicted_quality}** ⚠️")

                with st.expander("Lihat Tingkat Kepercayaan Model"):
                    proba_df = pd.DataFrame(prediction_proba, columns=quality_labels.values())
                    st.bar_chart(proba_df.T.rename(columns={0:'Probabilitas'}))
            else:
                st.error("Tidak ada data untuk dianalisis. Silakan pilih mode input yang valid.")

# --- HALAMAN 2: DASHBOARD & DATA ---
elif page == "Dashboard & Data":
    st.title("📊 Dashboard & Eksplorasi Data")
    st.markdown("Halaman ini untuk menganalisis dan menjelajahi dataset `WQD.xlsx`.")

    if full_data is not None:
        st.sidebar.header("Filter Data")
        quality_filter = st.sidebar.selectbox(
            "Tampilkan data berdasarkan kualitas:",
            options=["Semua", "Excellent", "Good", "Poor"],
            index=0
        )

        if quality_filter == "Semua":
            filtered_data = full_data
        else:
            filtered_data = full_data[full_data['Quality Label'] == quality_filter]

        st.write(f"Menampilkan **{len(filtered_data)}** data dengan kualitas: **{quality_filter}**")

        st.subheader("📋 Tabel Data")
        st.dataframe(filtered_data, use_container_width=True)

        st.subheader("📈 Visualisasi Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi Kategori Kualitas Air**")
            quality_counts = full_data['Quality Label'].value_counts()
            fig_pie = px.pie(
                names=quality_counts.index,
                values=quality_counts.values,
                title="Proporsi Kategori"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("**Distribusi Parameter per Kategori**")
            params_to_plot = full_data.drop(columns=['Water Quality', 'Quality Label']).columns.tolist()
            selected_param = st.selectbox("Pilih Parameter:", params_to_plot)
            
            fig_hist = px.histogram(
                full_data,
                x=selected_param,
                color='Quality Label',
                barmode='overlay',
                title=f"Distribusi '{selected_param}'"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("**Statistik Deskriptif per Kategori**")
        stats_data = full_data.drop(columns=['Quality Label'])
        desc_stats = stats_data.groupby('Water Quality').describe().T
        st.dataframe(desc_stats, use_container_width=True)