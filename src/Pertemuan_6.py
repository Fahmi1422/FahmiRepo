# src/pertemuan_6.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- BAGIAN INI SUDAH DIPERBAIKI ---
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer # Lokasi SimpleImputer yang Benar!
# ------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

# Konfigurasi Direktori & Variabel
PROCESSED_PATH = 'dataset/processed_kelulusan.csv'
MODEL_PATH = 'model/rf_model.pkl' 
RESULT_PATH = 'result'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

# Definisikan random state untuk reproduksibilitas
RANDOM_STATE = 42
TARGET_COL = "Lulus"
TEST_SIZE_FRAC = 0.3

print("--- Memulai Pertemuan 6: Random Forest untuk Klasifikasi ---")

## Langkah 1 — Muat Data & Split Ulang (Pilihan A)
try:
    df = pd.read_csv(PROCESSED_PATH)
except FileNotFoundError:
    print(f"ERROR: File {PROCESSED_PATH} tidak ditemukan. Pastikan Anda sudah menjalankan pertemuan_4.py.")
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
print(f"Data Shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


## Langkah 2 — Pipeline & Baseline Random Forest
print("\n[Langkah 2: Pipeline & Baseline Random Forest]")

# Setup Preprocessing Pipeline
num_cols = X_train.select_dtypes(include="number").columns
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

# Model Baseline RF
rf_baseline = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE
)
pipe_baseline = Pipeline([("pre", preprocessor), ("clf", rf_baseline)])

# Training & Validation
pipe_baseline.fit(X_train, y_train)
y_val_pred_base = pipe_baseline.predict(X_val)
f1_base_val = f1_score(y_val, y_val_pred_base, average="macro")

print(f"Baseline RF — F1-Macro (Validation Set): {f1_base_val:.4f}")
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_base, digits=3))


## Langkah 3 — Validasi Silang (Baseline Performance)
print("\n[Langkah 3: Validasi Silang Baseline RF (Train Set)]")

# Setup Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe_baseline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)

print(f"CV F1-Macro (Train Set): {scores.mean():.4f} ± {scores.std():.4f}")


## Langkah 4 — Tuning Ringkas (GridSearch)
print("\n[Langkah 4: Tuning Ringkas (GridSearch)]")

# Hyperparameter Grid: Fokus pada kompleksitas pohon
param = {
  "clf__max_depth": [None, 8, 15], 
  "clf__min_samples_split": [2, 5], 
}

# Grid Search CV
gs = GridSearchCV(pipe_baseline, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=0) 
gs.fit(X_train, y_train)

print(f"Best params (CV): {gs.best_params_}")
print(f"Best CV F1-Macro Score: {gs.best_score_:.4f}")

best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
f1_best_val = f1_score(y_val, y_val_best, average="macro")
print(f"Best Tuned RF — F1-Macro (Validation Set): {f1_best_val:.4f}")

# Final Model Selection 
final_model = best_model if f1_best_val >= f1_base_val else pipe_baseline
model_name = "Tuned Random Forest" if final_model is best_model else "Baseline Random Forest"
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

# ROC-AUC dan Kurva
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    
    # 5a. ROC-AUC dan ROC Curve
    try:
        auc_score = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC-AUC (Test Set): {auc_score:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5)); 
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})');
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); 
        plt.title("ROC Curve (Test Set)")
        plt.legend(loc="lower right")
        plt.tight_layout(); 
        plt.savefig(os.path.join(RESULT_PATH, "p6_roc_test.png"), dpi=120)
        print(f"ROC Curve telah disimpan ke {RESULT_PATH}/p6_roc_test.png")
        
    except ValueError as e:
        print(f"Tidak dapat menghitung ROC-AUC karena: {e}")

    # 5b. Precision-Recall (PR) Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5)); 
    plt.plot(rec, prec); 
    plt.xlabel("Recall"); 
    plt.ylabel("Precision"); 
    plt.title("PR Curve (Test Set)")
    plt.tight_layout(); 
    plt.savefig(os.path.join(RESULT_PATH, "p6_pr_test.png"), dpi=120)
    print(f"PR Curve telah disimpan ke {RESULT_PATH}/p6_pr_test.png")


## Langkah 6 — Pentingnya Fitur (Feature Importance)
print("\n[Langkah 6: Pentingnya Fitur (Feature Importance)]")

# 6a) Feature importance native (gini)
try:
    feature_names = final_model.named_steps["pre"].get_feature_names_out(input_features=X.columns)
    importances = final_model.named_steps["clf"].feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("Top feature importance (Gini-based):")
    for name, val in top_features:
        print(f"{name.split('__')[1]}: {val:.4f}") 
        
    # Visualisasi Feature Importance
    names = [item[0].split('__')[1] for item in top_features]
    values = [item[1] for item in top_features]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=values, y=names)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "p6_feature_importance.png"), dpi=120)
    print(f"Visualisasi Feature Importance disimpan di {RESULT_PATH}/p6_feature_importance.png")
    
except Exception as e:
    print(f"Feature importance tidak dapat ditampilkan: {e}")


## Langkah 7 — Simpan Model
print("\n[Langkah 7: Simpan Model]")
joblib.dump(final_model, MODEL_PATH)
print(f"Model tersimpan ke {MODEL_PATH}")


## Langkah 8 — Cek Inference Lokal
print("\n[Langkah 8: Cek Inference Lokal (Contoh Fiktif)]")

sample_data = {
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}
sample = pd.DataFrame([sample_data])

try:
    mdl = joblib.load(MODEL_PATH)
    prediction = int(mdl.predict(sample)[0])

    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(sample)[0, 1]
        print(f"Prediksi Lulus: {prediction} (Probabilitas Lulus: {proba:.4f})")
    else:
        print(f"Prediksi Lulus: {prediction}")
except FileNotFoundError:
    print(f"Gagal memuat model dari {MODEL_PATH}.")
    

print("\n--- Pertemuan 6 Selesai ---")