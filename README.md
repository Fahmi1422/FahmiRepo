🎓 Klasifikasi Kelulusan Mahasiswa (P4 - P7)
Proyek Machine Learning untuk memprediksi kelulusan mahasiswa. Meliputi Data Preparation (P4), Ensemble Modeling (Random Forest, P6), dan Deep Learning (ANN, P7).

🚀 Ringkasan Kinerja Model
Kami membandingkan model Random Forest (RF) dan Artificial Neural Network (ANN) untuk klasifikasi biner (Lulus: 1 vs Tidak Lulus: 0).
Model
Fokus Utama
Metrik AUC (Test Set)
Random Forest (P6)
Feature Importance (Prediksi Berbasis Bobot Fitur)
[Isi Nilai AUC Anda]
ANN (P7)
Regularisasi (Dropout & Early Stopping)
[Isi Nilai AUC Anda]

📁 Struktur dan Reproduksi
Semua code terletak di folder src/. Output model dan plot disimpan di model/ dan result/.
Cara Menjalankan
Pastikan dependensi utama (tensorflow, scikit-learn, pandas) telah diinstal.
Jalankan skrip di folder src/ secara berurutan (P4 hingga P7) untuk mereproduksi seluruh pipeline Machine Learning.
Tips: Fitur IPK dan Waktu_Belajar_Jam teridentifikasi sebagai prediktor terkuat dalam model Random Forest.
