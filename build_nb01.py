"""Build Notebook 01: Preprocessing Sensor Data"""
import nbformat as nbf, os

BASE = r'd:\Dosen\0 Genap 2026\Penelitian\Pak Chindra\irigasi_ai'

def c(src): return nbf.v4.new_code_cell(src)
def m(src): return nbf.v4.new_markdown_cell(src)

cells = []

cells.append(m("""# 🌿 Notebook 01 — Preprocessing Sensor Data
**Penelitian AIoT Irigasi Adaptif Gambut – 3 Node ESP32**

Pipeline: Load DB → Analisis → Cleaning → Simpan `dataset/data_cleaned.csv`
"""))

cells.append(c("""\
# ── CELL 1: Import Library ─────────────────────────────────────
import sys, os, json, warnings
warnings.filterwarnings('ignore')

%pip install -q seaborn sqlalchemy pymysql

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

print('[OK] Library berhasil diimport')
print(f'   pandas : {pd.__version__}')
print(f'   numpy  : {np.__version__}')
"""))

cells.append(c("""\
# ── CELL 2: Koneksi Database & Load Data ───────────────────────
DB_URL = "mysql+pymysql://u832389661_unama1:Unama2026@153.92.15.62:3306/u832389661_unama1"
engine = create_engine(DB_URL)

query = "SELECT * FROM sensor_readings ORDER BY waktu_pembacaan ASC"
df_raw = pd.read_sql(query, engine)

print('=== 📊 Informasi Data Mentah ===')
print(f'Jumlah baris  : {df_raw.shape[0]:,}')
print(f'Jumlah kolom  : {df_raw.shape[1]}')
print(f'Kolom         : {list(df_raw.columns)}')
print(f'Dtype:\\n{df_raw.dtypes}')
print()
wmin = df_raw['waktu_pembacaan'].min()
wmax = df_raw['waktu_pembacaan'].max()
print(f'Periode data  : {wmin}  →  {wmax}')
print()
print(df_raw.head(3))
"""))

cells.append(c("""\
# ── CELL 3: Analisis Missing Values ────────────────────────────
print('=== 🔍 Analisis Missing Values ===')
miss    = df_raw.isnull().sum()
miss_pct= (miss / len(df_raw) * 100).round(3)
miss_df = pd.DataFrame({'Jumlah_Missing': miss, 'Persen(%)': miss_pct})
mask    = miss_df['Jumlah_Missing'] > 0
print(miss_df[mask] if mask.any() else '  [OK] Tidak ada missing value')
print(f'\\nTotal kolom tanpa missing: {(miss == 0).sum()}')
"""))

cells.append(c("""\
# ── CELL 4: Analisis Nilai Nol & Negatif ───────────────────────
print('=== ⚠️  Analisis Nilai Nol/Negatif ===')

sensor_cols = [
    'kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3',
    'kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3',
    'suhu_udara_1','suhu_udara_2','suhu_udara_3',
    'tekanan_udara_1','tekanan_udara_2','tekanan_udara_3',
    'curah_hujan'
]
sensor_cols = [c for c in sensor_cols if c in df_raw.columns]

zero_df = pd.DataFrame({
    'Jumlah_Nol' : (df_raw[sensor_cols] == 0).sum(),
    'Persen(%)' : ((df_raw[sensor_cols] == 0).sum() / len(df_raw) * 100).round(3)
})
print(zero_df)

print('\\n=== Nilai Negatif pada Tekanan Udara ===')
for col in ['tekanan_udara_1','tekanan_udara_2','tekanan_udara_3']:
    if col in df_raw.columns:
        n = (df_raw[col] < 0).sum()
        print(f'  {col}: {n} nilai negatif')
"""))

cells.append(c("""\
# ── CELL 5: Visualisasi Distribusi Data Mentah ─────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('📊 Distribusi Sensor Data Mentah per Node', fontsize=14, fontweight='bold')

groups = [
    (['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3'], 'Kelembaban Tanah (%)'),
    (['kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3'],  'Kelembaban Udara (%)'),
    (['suhu_udara_1','suhu_udara_2','suhu_udara_3'],                    'Suhu Udara (°C)'),
]
colors = ['#2196F3','#4CAF50','#FF9800']

for row, (cols, ylabel) in enumerate(groups):
    for col_idx, col in enumerate(cols):
        if col in df_raw.columns:
            ax = axes[row][col_idx]
            ax.hist(df_raw[col].dropna(), bins=50, color=colors[col_idx], alpha=0.7, edgecolor='white')
            ax.set_title(f'Node {col_idx+1}', fontweight='bold')
            ax.set_xlabel(ylabel)

plt.tight_layout()
os.makedirs('dataset', exist_ok=True)
plt.savefig('dataset/plot_distribusi_raw.png', dpi=150, bbox_inches='tight')
plt.show()
print('[OK] Plot distribusi raw disimpan')
"""))

cells.append(c("""\
# ── CELL 6: Data Cleaning ──────────────────────────────────────
df = df_raw.copy()
print(f'Baris awal: {len(df):,}')

# 6a. Konversi dan validasi timestamp
df['waktu_pembacaan'] = pd.to_datetime(df['waktu_pembacaan'], errors='coerce')
n_invalid_time = df['waktu_pembacaan'].isna().sum()
if n_invalid_time:
    print(f'  Hapus {n_invalid_time} baris timestamp tidak valid')
df = df.dropna(subset=['waktu_pembacaan'])
df = df.sort_values('waktu_pembacaan').reset_index(drop=True)

# 6b. Hapus duplikat timestamp (ambil yang pertama)
n_dup = df.duplicated(subset=['waktu_pembacaan']).sum()
df = df.drop_duplicates(subset=['waktu_pembacaan'], keep='first').reset_index(drop=True)
print(f'  Hapus {n_dup} baris duplikat timestamp => {len(df):,} baris')

# 6c. Hapus kolom tidak diperlukan untuk ML
drop_cols = ['id','backup_status','created_at','updated_at',
             'sensor_tanah_adc_1','sensor_tanah_adc_2','sensor_tanah_adc_3']
dropped = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=dropped)
print(f'  Kolom dihapus: {dropped}')

# 6d. Pastikan semua sensor kolom numerik
num_cols = [col for col in df.columns if col != 'waktu_pembacaan']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# 6e. Hapus baris jika SEMUA kelembaban tanah = 0 (sensor mati)
kt_cols = [col for col in ['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3']
           if col in df.columns]
if kt_cols:
    mask_all_zero = (df[kt_cols] == 0).all(axis=1)
    n_allzero = int(mask_all_zero.sum())
    df = df[~mask_all_zero].copy().reset_index(drop=True)
    print(f'  Hapus {n_allzero} baris semua sensor tanah = 0')

# 6f. Interpolasi nilai 0 pada kelembaban tanah (sensor gagal baca sementara)
# Gunakan method='linear' — index integer bukan DatetimeIndex
for col in kt_cols:
    n_zero = int((df[col] == 0).sum())
    df[col] = df[col].replace(0, np.nan).interpolate(method='linear')
    df[col] = df[col].bfill().ffill()
    print(f'  Interpolasi {n_zero} nilai 0 pada {col}')

# 6g. Bersihkan tekanan udara tidak valid (< 800 hPa atau > 1100 hPa)
for col in ['tekanan_udara_1','tekanan_udara_2','tekanan_udara_3']:
    if col in df.columns:
        mask_inv = (df[col] < 800) | (df[col] > 1100)
        n_inv = int(mask_inv.sum())
        df.loc[mask_inv, col] = np.nan
        df[col] = df[col].interpolate(method='linear').bfill().ffill()
        print(f'  Bersihkan {n_inv} nilai tekanan tidak valid pada {col}')

# 6h. Bersihkan kelembaban udara > 100%
for col in ['kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3']:
    if col in df.columns:
        mask_inv = df[col] > 100
        n_inv = int(mask_inv.sum())
        df.loc[mask_inv, col] = np.nan
        df[col] = df[col].interpolate(method='linear').bfill().ffill()
        if n_inv:
            print(f'  Bersihkan {n_inv} kelembaban udara > 100% pada {col}')

# 6i. Hapus outlier ekstrem (batas 1%-99% persentil, IQR x 3.0)
print('\\n=== Hapus Outlier Ekstrem ===')
for col in kt_cols:
    Q1   = df[col].quantile(0.01)
    Q3   = df[col].quantile(0.99)
    IQR  = Q3 - Q1
    low  = max(Q1 - 3.0 * IQR, 0.0)
    high = min(Q3 + 3.0 * IQR, 100.0)
    mask_out = (df[col] < low) | (df[col] > high)
    n_out = int(mask_out.sum())
    df.loc[mask_out, col] = np.nan
    df[col] = df[col].interpolate(method='linear').bfill().ffill()
    print(f'  {col}: batas [{low:.1f}, {high:.1f}] => {n_out} nilai diinterpolasi')

print(f'\\nBaris akhir setelah cleaning: {len(df):,}')
print(f'Kolom akhir : {list(df.columns)}')
"""))

cells.append(c("""\
# ── CELL 7: Verifikasi Hasil Cleaning ──────────────────────────
print('=== [OK] Statistik Data Setelah Cleaning ===')
df_stat = df.describe().T[['count','mean','std','min','max']]
print(df_stat.to_string())

print(f'\\nMissing values tersisa:')
print(df.isnull().sum()[df.isnull().sum() > 0])
"""))

cells.append(c("""\
# ── CELL 8: Visualisasi Time Series Setelah Cleaning ───────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle('📈 Time Series Kelembaban Tanah per Node (Setelah Cleaning)',
             fontsize=14, fontweight='bold')

colors_ts = ['#1565C0','#2E7D32','#E65100']
for i, (col, color) in enumerate(
        zip(['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3'], colors_ts)):
    if col in df.columns:
        axes[i].plot(df['waktu_pembacaan'], df[col],
                     color=color, linewidth=0.5, alpha=0.8)
        axes[i].set_ylabel('Kelembaban (%)')
        axes[i].set_title(f'Node {i+1} — {col}', fontweight='bold')
        axes[i].set_ylim(0, 105)
        axes[i].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Threshold 60%')
        axes[i].legend(fontsize=8)

axes[-1].set_xlabel('Waktu Pembacaan')
plt.tight_layout()
plt.savefig('dataset/plot_timeseries_cleaned.png', dpi=150, bbox_inches='tight')
plt.show()
print('[OK] Plot time series disimpan')
"""))

cells.append(c("""\
# ── CELL 9: Simpan Dataset Bersih ──────────────────────────────
os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/data_cleaned.csv', index=False)

# Simpan cleaning report
report = {
    'baris_awal'  : int(df_raw.shape[0]),
    'baris_akhir' : int(len(df)),
    'kolom_akhir' : list(df.columns),
    'kolom_dihapus': ['id','backup_status','created_at','updated_at',
                       'sensor_tanah_adc_1','sensor_tanah_adc_2','sensor_tanah_adc_3'],
    'periode_awal' : str(df['waktu_pembacaan'].min()),
    'periode_akhir': str(df['waktu_pembacaan'].max()),
}
with open('dataset/cleaning_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('=== 💾 Dataset Bersih Tersimpan ===')
print(f'File   : dataset/data_cleaned.csv')
print(f'Baris  : {len(df):,}')
print(f'Kolom  : {df.shape[1]}')
print(f'Periode: {df["waktu_pembacaan"].min()} → {df["waktu_pembacaan"].max()}')
print()
print('[OK] Preprocessing selesai! Lanjut ke → 02_preprocessing.ipynb')
"""))

# Build notebook
notebook = nbf.v4.new_notebook()
notebook['cells'] = cells
notebook['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'}
}

out = os.path.join(BASE, '01_preprocessing_sensor.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(notebook, f)
print(f'[OK] Notebook 01 berhasil dibuat: {out}')