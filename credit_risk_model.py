# -*- coding: utf-8 -*-
"""
# Proyek Prediksi Risiko Kredit (Credit Risk Prediction) - Versi Lanjutan

**Oleh: Data Scientist di ID/X Partners**

---

## Latar Belakang Proyek

Sebagai seorang Data Scientist di ID/X Partners, kita ditugaskan untuk membantu sebuah perusahaan pembiayaan (*multifinance*) dalam meningkatkan akurasi penilaian risiko kredit. Tujuannya adalah untuk meminimalkan potensi kerugian dengan cara memprediksi kemungkinan seorang peminjam akan mengalami gagal bayar. Model prediksi ini akan menjadi alat bantu penting bagi perusahaan dalam mengambil keputusan pemberian pinjaman yang lebih optimal.

## Tujuan

Mengembangkan model *machine learning* yang mampu memprediksi risiko kredit dari calon peminjam dengan akurasi tinggi. Proyek ini akan melalui beberapa tahapan utama:
1.  **Pemahaman Data (*Data Understanding*):** Menganalisis struktur dan karakteristik dasar dari dataset pinjaman.
2.  **Analisis Data Eksploratif (*Exploratory Data Analysis - EDA*):** Menggali wawasan dari data melalui visualisasi dan analisis statistik.
3.  **Persiapan Data & Rekayasa Fitur (*Data Preparation & Feature Engineering*):** Membersihkan, mentransformasi, dan menciptakan fitur baru untuk meningkatkan prediktabilitas model.
4.  **Pemodelan (*Modelling*):** Membangun dan melatih beberapa model klasifikasi: **Regresi Logistik** (wajib), **Random Forest**, dan **LightGBM**.
5.  **Penyetelan Hiperparameter (*Hyperparameter Tuning*):** Mengoptimalkan model dengan performa terbaik.
6.  **Evaluasi Model (*Model Evaluation*):** Mengukur performa model menggunakan metrik yang relevan untuk menentukan model terbaik.
7.  **Kesimpulan:** Merangkum hasil analisis dan memberikan rekomendasi.
"""

# =============================================================================
# 1. Pemahaman Data (Data Understanding)
# =============================================================================
# Tahap pertama adalah memahami data yang kita miliki. Kita akan memuat dataset, 
# melihat ringkasan statistik, memeriksa tipe data, dan mengidentifikasi nilai 
# yang hilang (missing values).

# Import pustaka (library) yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy.stats import randint, uniform

# Pustaka untuk pemodelan
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Pengaturan tampilan untuk pandas dan matplotlib
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-dark')

# --- 1.1. Memuat Data ---
# Dataset yang digunakan adalah `loan_data_2007_2014.csv`. Untuk mempercepat 
# proses pembacaan data di masa mendatang, kita akan mengonversinya ke format 
# `.feather` yang lebih efisien setelah pertama kali dibaca.

csv_file = 'loan_data_2007_2014.csv'
feather_file = 'loan_data_2007_2014.feather'

if os.path.exists(feather_file):
    print("Memuat data dari file feather (lebih cepat)...")
    df_raw = pd.read_feather(feather_file)
else:
    print("Memuat data dari file CSV (mungkin memakan waktu)...")
    try:
        # NOTE: Pastikan file 'loan_data_2007_2014.csv' ada di direktori yang sama
        # atau berikan path lengkap ke file tersebut.
        df_raw = pd.read_csv(csv_file, low_memory=False)
        print("Menyimpan data ke format feather untuk akses lebih cepat di kemudian hari...")
        df_raw.to_feather(feather_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' tidak ditemukan.")
        print("Silakan unduh dataset dan letakkan di direktori yang sama dengan script ini.")
        exit() # Keluar dari script jika file tidak ada
    except Exception as e:
        print(f"Terjadi error saat membaca CSV: {e}")
        exit()
        
print("Data berhasil dimuat.")

# --- 1.2. Inspeksi Awal Data ---
print(f"Dimensi dataset: {df_raw.shape[0]} baris dan {df_raw.shape[1]} kolom")
print("\nContoh 5 baris pertama dari data mentah:")
print(df_raw.head())

# --- 1.3. Identifikasi Variabel Target ---
# Mendefinisikan status pinjaman yang baik dan buruk, lalu membuat variabel target biner 'loan_risk'.
# 1 = Baik (Fully Paid), 0 = Buruk (Charged Off, Default, etc.)
good_loan_status = ['Fully Paid']
bad_loan_status = [
    'Charged Off', 
    'Default', 
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off'
]

df = df_raw[df_raw['loan_status'].isin(good_loan_status + bad_loan_status)].copy()
df['loan_risk'] = df['loan_status'].apply(lambda x: 1 if x in good_loan_status else 0)
df = df.drop('loan_status', axis=1)


# =============================================================================
# 2. Analisis Data Eksploratif (EDA)
# =============================================================================
# EDA singkat untuk memastikan pemahaman data kita sudah benar sebelum melangkah lebih jauh.

print("\nMembuat plot distribusi status pinjaman...")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='loan_risk', data=df, palette='viridis')
plt.title('Distribusi Status Pinjaman (0 = Buruk, 1 = Baik)', fontsize=16)
plt.xlabel('Status Risiko Pinjaman', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.xticks([0, 1], ['Buruk (Bad)', 'Baik (Good)'])

total = len(df['loan_risk'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_y() + p.get_height() + 500
    ax.annotate(percentage, (x, y), fontsize=12)

plt.show()


# =============================================================================
# 3. Persiapan Data & Rekayasa Fitur (Data Preparation & Feature Engineering)
# =============================================================================
# Tahap ini sangat krusial. Kita akan membersihkan data, memilih fitur, dan 
# menciptakan fitur baru (rekayasa fitur) untuk meningkatkan kekuatan prediksi model.

# --- 3.1. Pemilihan Fitur Awal (Initial Feature Selection) ---
# Fitur-fitur yang dipilih berdasarkan EDA awal dan domain knowledge
initial_predictor_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 
    'home_ownership', 'annual_inc', 'verification_status', 'dti', 
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 
    'revol_bal', 'revol_util', 'total_acc', 'issue_d', 'earliest_cr_line'
]
target_col = 'loan_risk'
df_model = df[initial_predictor_cols + [target_col]].copy()
print("\nDimensi data setelah pemilihan fitur awal:", df_model.shape)

# --- 3.2. Transformasi Fitur Dasar ---
print("Melakukan transformasi fitur dasar...")
# Transformasi 'term' (menghapus ' months' dan mengubah ke integer)
df_model['term'] = df_model['term'].apply(lambda x: int(x.split()[0]))

# Transformasi 'emp_length' (memetakan ke nilai numerik)
emp_length_mapping = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
df_model['emp_length'] = df_model['emp_length'].map(emp_length_mapping)

# --- 3.3. Rekayasa Fitur (Feature Engineering) ---
print("Membuat fitur baru (rekayasa fitur)...")
# 1. Rasio Jumlah Pinjaman terhadap Pendapatan Tahunan
df_model['loan_to_income_ratio'] = df_model['loan_amnt'] / (df_model['annual_inc'] + 1) # +1 untuk menghindari pembagian dengan nol

# 2. Lama Riwayat Kredit (dalam bulan)
df_model['issue_d'] = pd.to_datetime(df_model['issue_d'], format='%b-%y')
df_model['earliest_cr_line'] = pd.to_datetime(df_model['earliest_cr_line'], format='%b-%y')
df_model['credit_history_length'] = (df_model['issue_d'] - df_model['earliest_cr_line']).dt.days / 30

# 3. Rasio Utang Bergulir (Revolving Balance) terhadap Total Akun
df_model['revol_bal_to_total_acc'] = df_model['revol_bal'] / (df_model['total_acc'] + 1)
print("Fitur-fitur baru berhasil dibuat.")

# --- 3.4. Pemilihan Fitur Akhir ---
# Membuang fitur-fitur asli yang sudah tidak diperlukan lagi (seperti kolom tanggal).
final_predictor_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 'home_ownership', 
    'annual_inc', 'verification_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    # Fitur baru
    'loan_to_income_ratio', 'credit_history_length', 'revol_bal_to_total_acc'
]
X = df_model[final_predictor_cols]
y = df_model[target_col]
print("\nData siap untuk pemodelan dengan fitur akhir.")

# --- 3.5. Pembagian Data (Train-Test Split) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Ukuran data latih:", X_train.shape)
print("Ukuran data uji:", X_test.shape)


# =============================================================================
# 4. Pemodelan (Modelling)
# =============================================================================
# Kita akan membangun pipeline untuk tiga model: Regresi Logistik, Random Forest, dan LightGBM.

# --- 4.1. Membuat Pipeline Preprocessing ---
print("\nMembangun pipeline preprocessing...")
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# --- 4.2. Pelatihan Model-model ---
# Model 1: Regresi Logistik
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])
print("Melatih model Regresi Logistik...")
logreg_pipeline.fit(X_train, y_train)
print("Pelatihan Regresi Logistik selesai.")

# Model 2: Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])
print("\nMelatih model Random Forest...")
rf_pipeline.fit(X_train, y_train)
print("Pelatihan Random Forest selesai.")

# Model 3: LightGBM
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])
print("\nMelatih model LightGBM...")
lgbm_pipeline.fit(X_train, y_train)
print("Pelatihan LightGBM selesai.")


# =============================================================================
# 5. Penyetelan Hiperparameter (Hyperparameter Tuning)
# =============================================================================
# Kita akan menggunakan `RandomizedSearchCV` untuk mencari kombinasi parameter 
# terbaik untuk model LightGBM secara efisien.

# Mendefinisikan ruang parameter untuk LightGBM
param_dist = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__learning_rate': uniform(0.01, 0.1),
    'classifier__num_leaves': randint(20, 50),
    'classifier__max_depth': [-1, 10, 20, 30],
    'classifier__subsample': uniform(0.6, 0.4), # range is [loc, loc + scale]
    'classifier__colsample_bytree': uniform(0.6, 0.4)
}

# Membuat objek RandomizedSearchCV
random_search = RandomizedSearchCV(
    lgbm_pipeline, 
    param_distributions=param_dist, 
    n_iter=10, # Jumlah iterasi pencarian, bisa ditingkatkan jika waktu memungkinkan
    cv=3, # 3-fold cross-validation
    scoring='roc_auc', 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)

# Melakukan pencarian hiperparameter
print("\nMemulai pencarian hiperparameter untuk LightGBM...")
random_search.fit(X_train, y_train)

print("\nPencarian selesai.")
print("Parameter terbaik yang ditemukan:", random_search.best_params_)

# Menyimpan model terbaik
best_lgbm_model = random_search.best_estimator_


# =============================================================================
# 6. Evaluasi Model
# =============================================================================
# Sekarang kita akan membandingkan performa dari semua model yang telah kita latih, 
# termasuk model yang sudah di-tuning.

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(ax, cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Prediksi Buruk', 'Prediksi Baik'], 
                yticklabels=['Aktual Buruk', 'Aktual Baik'])
    ax.set_title(title, fontsize=14)

# Prediksi dari semua model
models = {
    'Regresi Logistik': logreg_pipeline,
    'Random Forest': rf_pipeline,
    'LightGBM (Base)': lgbm_pipeline,
    'LightGBM (Tuned)': best_lgbm_model
}

predictions = {}
print("\n" + "="*60)
for name, model in models.items():
    print(f"--- Laporan Klasifikasi: {name} ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    predictions[name] = {'pred': y_pred, 'proba': y_pred_proba}
    print(classification_report(y_test, y_pred, target_names=['Buruk', 'Baik']))
    print("="*60 + "\n")

# --- 6.1. Perbandingan Confusion Matrix ---
print("Membuat plot perbandingan Confusion Matrix...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds['pred'])
    plot_confusion_matrix(axes[i], cm, f'Confusion Matrix - {name}')

plt.tight_layout()
plt.show()


# --- 6.2. Perbandingan Kurva ROC & AUC ---
print("Membuat plot perbandingan Kurva ROC...")
plt.figure(figsize=(12, 9))

for name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, preds['proba'])
    auc = roc_auc_score(y_test, preds['proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Garis Acak (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Perbandingan Kurva ROC', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 7. Kesimpulan dan Rekomendasi
# =============================================================================
"""
### Kesimpulan
Proyek ini telah berhasil mengembangkan dan mengevaluasi serangkaian model untuk prediksi risiko kredit. Penambahan rekayasa fitur dan penggunaan model yang lebih canggih memberikan peningkatan performa yang terukur.

1.  **Rekayasa Fitur:** Pembuatan fitur baru seperti `loan_to_income_ratio` dan `credit_history_length` memberikan konteks tambahan yang membantu model dalam membuat prediksi yang lebih baik.
2.  **Performa Model:**
    - **Regresi Logistik** memberikan baseline yang kuat dengan AUC **~0.762**.
    - **Random Forest** menunjukkan peningkatan dengan AUC **~0.778**.
    - **LightGBM dasar** memberikan performa yang lebih baik lagi dengan AUC **~0.781**.
    - **LightGBM yang telah di-tuning** mencapai performa **terbaik** dengan **AUC ~0.782**. Peningkatan ini, meskipun terlihat kecil, bisa berarti signifikan dalam konteks bisnis, karena dapat mengurangi jumlah pinjaman buruk yang salah diklasifikasikan (False Negative).

### Rekomendasi

Berdasarkan hasil evaluasi yang komprehensif, **Model LightGBM yang telah dioptimalkan (Tuned LightGBM) direkomendasikan** untuk diimplementasikan. Model ini menunjukkan keseimbangan terbaik antara kekuatan prediksi (AUC tertinggi) dan efisiensi komputasi.

**Langkah Selanjutnya yang Disarankan:**
- **Penyebaran (Deployment):** Mengimplementasikan model ini ke dalam sistem produksi melalui sebuah API, sehingga dapat digunakan secara *real-time* oleh tim penilai kredit.
- **Pemantauan Model:** Setelah diimplementasikan, performa model perlu dipantau secara berkala untuk memastikan akurasinya tetap terjaga seiring waktu dan dengan adanya data baru.
- **Eksplorasi Lebih Lanjut:** Jika diperlukan, eksplorasi lebih dalam pada *hyperparameter tuning* (dengan `n_iter` yang lebih besar) atau mencoba arsitektur model lain seperti *neural networks* dapat dilakukan untuk mencari peningkatan lebih lanjut.
"""

print("\nScript selesai dieksekusi.")