# src/pertemuan_4.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Konfigurasi Direktori
# PASTIKAN DATASET BARU (50 BARIS) SUDAH DISIMPAN DI FILE INI
DATASET_PATH = 'dataset/kelulusan_mahasiswa.csv'
PROCESSED_PATH = 'dataset/processed_kelulusan.csv'
RESULT_PATH = 'result'

# Pastikan folder ada
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


print("--- Memulai Pertemuan 4: Data Preparation ---")

## Langkah 2 — Collection
print("\n[Langkah 2: Collection]")
try:
    df = pd.read_csv(DATASET_PATH)
    print("DataFrame Info:")
    df.info()
    print("\nDataFrame Head:")
    print(df.head())
except FileNotFoundError:
    print(f"ERROR: File {DATASET_PATH} tidak ditemukan. Pastikan Anda sudah membuatnya.")
    exit()

## Langkah 3 — Cleaning
print("\n[Langkah 3: Cleaning]")

# 3a. Missing Value Check (Seharusnya nol setelah data simulasi)
print("Missing Values per Kolom:")
print(df.isnull().sum())

# 3b. Hapus Duplikasi
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Jumlah baris awal: {initial_rows}, Setelah hapus duplikasi: {len(df)}")

# 3c. Identifikasi Outlier (Menggunakan Boxplot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.boxplot(y=df['IPK']).set_title('Boxplot IPK')
plt.subplot(1, 3, 2)
sns.boxplot(y=df['Jumlah_Absensi']).set_title('Boxplot Absensi')
plt.subplot(1, 3, 3)
sns.boxplot(y=df['Waktu_Belajar_Jam']).set_title('Boxplot Waktu Belajar')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'p4_boxplots_outlier.png'))
print("Boxplots outlier telah disimpan di result/p4_boxplots_outlier.png.")


## Langkah 4 — Exploratory Data Analysis (EDA)
print("\n[Langkah 4: EDA]")

# 4a. Statistik Deskriptif
print("Statistik Deskriptif:")
print(df.describe())

# 4b. Histogram Distribusi IPK
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['IPK'], bins=10, kde=True).set_title('Distribusi IPK') 
plt.subplot(1, 3, 2)
sns.histplot(df['Jumlah_Absensi'], bins=10, kde=True).set_title('Distribusi Absensi')
plt.subplot(1, 3, 3)
sns.histplot(df['Waktu_Belajar_Jam'], bins=10, kde=True).set_title('Distribusi Waktu Belajar')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'p4_histograms.png'))
print("Histograms distribusi telah disimpan di result/p4_histograms.png.")

# 4c. Scatterplot (IPK vs Waktu Belajar) dengan hue Lulus
plt.figure(figsize=(6, 5))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', style='Lulus', s=100)
plt.title('IPK vs Waktu Belajar (Berwarna Label Lulus)')
plt.savefig(os.path.join(RESULT_PATH, 'p4_scatterplot_ipk_study.png'))
print("Scatterplot IPK vs Waktu Belajar telah disimpan di result/p4_scatterplot_ipk_study.png.")

# 4d. Heatmap Korelasi
plt.figure(figsize=(6, 5))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Heatmap Korelasi')
plt.savefig(os.path.join(RESULT_PATH, 'p4_heatmap_korelasi.png'))
print("Heatmap korelasi telah disimpan di result/p4_heatmap_korelasi.png.")


## Langkah 5 — Feature Engineering
print("\n[Langkah 5: Feature Engineering]")

# Fitur Turunan 1: Rasio Absensi terhadap Batas Maksimal (Asumsi Batas Maksimal = 14)
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14

# Fitur Turunan 2: Interaksi IPK * Waktu_Belajar_Jam
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

print("\nFitur Baru (IPK_x_Study & Rasio_Absensi) telah ditambahkan:")
print(df.head())

# Simpan data yang sudah diproses
df.to_csv(PROCESSED_PATH, index=False)
print(f"\nData bersih dan fitur baru telah disimpan ke: {PROCESSED_PATH}")


## Langkah 6 — Splitting Dataset (Stratified Split) - PERBAIKAN DENGAN DATASET BESAR
print("\n[Langkah 6: Splitting Dataset (Train 70%, Validation 15%, Test 15%)]")

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Tahap 1: Split Train (70%) dan Temp (30%) - Menggunakan Stratified Split
# Error teratasi karena data temp (30% dari 50 = 15 baris) memiliki minimal 2 anggota per kelas.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Tahap 2: Split Temp (30%) menjadi Validation (15%) dan Test (15%) - Menggunakan Stratified Split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42) # stratify=y_temp sekarang aman

# Cek hasil splitting
print("\nPembagian Dataset:")
print(f"Train Set Shape: {X_train.shape}")
print(f"Validation Set Shape: {X_val.shape}")
print(f"Test Set Shape: {X_test.shape}")

# Periksa proporsi label (stratified split harusnya menjaga proporsi)
total_lulus = y.sum()
total_tidak_lulus = len(y) - total_lulus
print(f"\nProporsi Label (Total) - Lulus: {total_lulus/len(y):.2f}, Tidak Lulus: {total_tidak_lulus/len(y):.2f}")
print(f"Proporsi Label (Train) - Lulus: {y_train.sum()/len(y_train):.2f}, Tidak Lulus: {(len(y_train)-y_train.sum())/len(y_train):.2f}")
print(f"Proporsi Label (Validation) - Lulus: {y_val.sum()/len(y_val):.2f}, Tidak Lulus: {(len(y_val)-y_val.sum())/len(y_val):.2f}")
print(f"Proporsi Label (Test) - Lulus: {y_test.sum()/len(y_test):.2f}, Tidak Lulus: {(len(y_test)-y_test.sum())/len(y_test):.2f}")

print("\n--- Data Preparation Selesai ---")