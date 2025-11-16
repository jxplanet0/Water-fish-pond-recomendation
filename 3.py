# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- 1. Muat Data ---
try:
    df = pd.read_excel('WQD.xlsx')
    # --- TAMBAHKAN BARIS INI ---
    # Ini akan membersihkan spasi di awal dan akhir setiap nama kolom
    df.columns = df.columns.str.strip()
    print("Data berhasil dimuat dan nama kolom telah dibersihkan.")
except FileNotFoundError:
    print("Error: File 'WQD.xlsx' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

# --- 2. Pemahaman dan Persiapan Data ---
# ... (kode sisanya tidak berubah) ...
if df['Water Quality'].dtype == 'object':
    quality_map = {'Excellent': 0, 'Good': 1, 'Poor': 2}
    df['Water Quality'] = df['Water Quality'].map(quality_map)
    print("Kolom 'Water Quality' telah dipetakan ke numerik.")

X = df.drop('Water Quality', axis=1)
y = df['Water Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data dibagi: {len(X_train)} sampel latih, {len(X_test)} sampel uji.")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Random Forest telah dilatih.")

accuracy = model.score(X_test, y_test)
print(f"Akurasi model pada data uji: {accuracy:.4f}")

joblib.dump(model, 'water_quality_model.pkl')
print("Model telah disimpan sebagai 'water_quality_model.pkl'")
