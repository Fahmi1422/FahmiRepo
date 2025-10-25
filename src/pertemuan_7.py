# src/pertemuan_7.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- Konfigurasi Direktori & Variabel ---
PROCESSED_PATH = 'dataset/processed_kelulusan.csv'
RESULT_PATH = 'result'
os.makedirs(RESULT_PATH, exist_ok=True)

# Definisikan random state untuk reproduksibilitas
RANDOM_STATE = 42
TARGET_COL = "Lulus"
TEST_SIZE_FRAC = 0.3

# Set seed untuk reproducibility di Keras/TF
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("--- Memulai Pertemuan 7: ANN untuk Klasifikasi ---")

## Langkah 1 — Siapkan Data (Scaling & Split)
print("\n[Langkah 1: Siapkan Data]")

try:
    df = pd.read_csv(PROCESSED_PATH)
except FileNotFoundError:
    print(f"ERROR: File {PROCESSED_PATH} tidak ditemukan. Jalankan pertemuan_4.py.")
    exit()

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Standard Scaler
sc = StandardScaler()
# Transformasi dilakukan pada seluruh dataset sebelum split (karena ini data tabular kecil)
Xs = sc.fit_transform(X) 
Xs = pd.DataFrame(Xs, columns=X.columns)

# Split 70/15/15 (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=TEST_SIZE_FRAC, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)

print(f"Data Shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


## Langkah 2 — Bangun Model ANN (MLP)
print("\n[Langkah 2: Bangun Model ANN]")

# Arsitektur: Input(5 fitur) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(16, ReLU) -> Output(1, Sigmoid)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],), name='Input_Layer'),
    layers.Dense(32, activation="relu", name='Hidden_1_Dense'),
    layers.Dropout(0.3, name='Dropout_1'),
    layers.Dense(16, activation="relu", name='Hidden_2_Dense'),
    layers.Dense(1, activation="sigmoid", name='Output_Sigmoid')  # Sigmoid untuk klasifikasi biner
])

# Kompilasi Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy", # Loss yang tepat untuk klasifikasi biner dengan Sigmoid
              metrics=["accuracy", "AUC"]) 
model.summary()


## Langkah 3 — Training dengan Early Stopping
print("\n[Langkah 3: Training dengan Early Stopping]")

# Early Stopping: Pantau val_loss, jika tidak ada perbaikan selama 10 epoch, hentikan.
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

# Training Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, # Epochs maksimal 100, akan dihentikan Early Stopping
    batch_size=32, # Batch size sesuai jumlah sampel train (32 baris)
    callbacks=[es], 
    verbose=1
)
print(f"Training selesai pada epoch: {len(history.history['loss'])}")


## Langkah 4 — Evaluasi di Test Set
print("\n[Langkah 4: Evaluasi di Test Set]")

# Evaluasi Loss dan Metrik
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

# Prediksi Probabilitas
y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int) # Hard prediction dengan threshold 0.5

# Classification Metrics
f1_final_test = f1_score(y_test, y_pred, average="macro")
print(f"Test F1-Macro Score: {f1_final_test:.4f}")

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, digits=3))


## Langkah 5 — Visualisasi Learning Curve
print("\n[Langkah 5: Visualisasi Learning Curve]")

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Learning Curve (ANN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
curve_path = os.path.join(RESULT_PATH, "p7_learning_curve.png")
plt.savefig(curve_path, dpi=120)
print(f"Learning Curve disimpan di {curve_path}")

print("\n--- Pertemuan 7 Selesai ---")

## Laporkan Hasil Eksperimen
print("\n[Laporan Hasil Eksperimen]")
print(f"Arsitektur: 5 -> 32(ReLU, Dropout 0.3) -> 16(ReLU) -> 1(Sigmoid)")
print(f"Optimizer: Adam (LR 1e-3)")
print(f"Regularisasi: Dropout, Early Stopping (Patience 10)")
print(f"Final Test Metrics: Accuracy={acc:.4f}, AUC={auc:.4f}, F1-Macro={f1_final_test:.4f}")