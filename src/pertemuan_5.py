# src/pertemuan_5.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Konfigurasi Direktori & Variabel
PROCESSED_PATH = 'dataset/processed_kelulusan.csv'
MODEL_PATH = 'model/model.pkl'
RESULT_PATH = 'result'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

# Definisikan random state untuk reproduksibilitas
RANDOM_STATE = 42
TARGET_COL = "Lulus"
TEST_SIZE_FRAC = 0.3

print("--- Memulai Pertemuan 5: Modeling & Validasi ---")

## Langkah 1 — Muat Data & Split Ulang
# Menggunakan Pilihan B dari Lembar Kerja: Memuat processed_kelulusan.csv lalu split ulang
try:
    df = pd.read_csv(PROCESSED_PATH)
except FileNotFoundError:
    print(f"ERROR: File {PROCESSED_PATH} tidak ditemukan. Jalankan pertemuan_4.py terlebih dahulu.")
    exit()

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Split 70% Train, 30% Temp (Temp = Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=TEST_SIZE_FRAC, stratify=y, random_state=RANDOM_STATE)
# Split Temp menjadi 15% Validation, 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)

print(f"\n[Langkah 1: Muat Data & Split]")
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


## Setup Pipeline Preprocessing
# Identifikasi kolom numerik (semua kolom selain 'Lulus')
num_cols = X_train.select_dtypes(include="number").columns

# Buat Preprocessing Pipeline: Imputasi Median + Scaling
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")


## Langkah 2 — Baseline Model: Logistic Regression
print("\n[Langkah 2: Baseline Model - Logistic Regression]")

# Selection: Logistic Regression adalah baseline yang baik, interpretable, dan cepat.
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
pipe_lr = Pipeline([("pre", preprocessor), ("clf", logreg)])

# Training & Validation
pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val)

# Metrik (Menggunakan F1-Macro karena label relatif seimbang 52%/48%, tapi F1 tetap baik untuk klasifikasi)
f1_lr_val = f1_score(y_val, y_val_pred_lr, average="macro")
print(f"Baseline (LogReg) F1-Macro (Validation Set): {f1_lr_val:.4f}")
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_lr, digits=3))


## Langkah 3 — Model Alternatif: Random Forest
print("\n[Langkah 3: Model Alternatif - Random Forest]")

# Selection: Random Forest (Tree-based model) bagus untuk menangani interaksi non-linear.
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE
)
pipe_rf = Pipeline([("pre", preprocessor), ("clf", rf)])

# Training & Validation
pipe_rf.fit(X_train, y_train)
y_val_pred_rf = pipe_rf.predict(X_val)
f1_rf_val = f1_score(y_val, y_val_pred_rf, average="macro")
print(f"RandomForest F1-Macro (Validation Set): {f1_rf_val:.4f}")


## Langkah 4 — Validasi Silang & Tuning Ringkas (Random Forest)
print("\n[Langkah 4: Validasi Silang & Tuning (Random Forest)]")

# Setup Cross-Validation (K-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Hyperparameter Grid untuk Tuning
param_grid = {
  "clf__max_depth": [None, 8, 15],  # Kedalaman Tree: [full, dangkal, menengah]
  "clf__min_samples_split": [2, 5], # Jumlah sampel minimum untuk membelah node
}

# Grid Search CV
gs = GridSearchCV(pipe_rf, param_grid=param_grid, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)

# Training (Fitting Grid Search ke Train Set)
gs.fit(X_train, y_train)

# Hasil Tuning
print(f"Best params (CV): {gs.best_params_}")
print(f"Best CV F1-Macro Score: {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
f1_best_rf_val = f1_score(y_val, y_val_best, average="macro")
print(f"Tuned RF F1-Macro (Validation Set): {f1_best_rf_val:.4f}")

# --- Pemilihan Model Final ---
if f1_best_rf_val >= f1_lr_val:
    final_model = best_rf
    model_name = "Tuned Random Forest"
else:
    final_model = pipe_lr
    model_name = "Logistic Regression (Baseline)"
print(f"\n>>> Model Final Terpilih: {model_name} <<<")


## Langkah 5 — Evaluasi Akhir (Test Set)
print("\n[Langkah 5: Evaluasi Akhir (Test Set)]")

y_test_pred = final_model.predict(X_test)

f1_final_test = f1_score(y_test, y_test_pred, average="macro")
print(f"F1-Macro (Test Set): {f1_final_test:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=3))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# ROC-AUC (jika ada predict_proba)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        auc_score = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC-AUC (Test Set): {auc_score:.4f}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5)); 
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})');
        plt.plot([0, 1], [0, 1], 'k--') # Diagonal line
        plt.xlabel("False Positive Rate"); 
        plt.ylabel("True Positive Rate"); 
        plt.title("ROC Curve (Test Set)")
        plt.legend(loc="lower right")
        plt.tight_layout(); 
        plt.savefig(os.path.join(RESULT_PATH, "p5_roc_test.png"), dpi=120)
        print(f"ROC Curve telah disimpan ke {RESULT_PATH}/p5_roc_test.png")
        
    except ValueError as e:
        print(f"Tidak dapat menghitung ROC-AUC karena: {e}")


## Langkah 6 — Simpan Model
print("\n[Langkah 6: Simpan Model]")
joblib.dump(final_model, MODEL_PATH)
print(f"Model tersimpan ke {MODEL_PATH}")

print("\n--- Modeling Pertemuan 5 Selesai ---")