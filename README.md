# Remote database
# ip : 153.92.15.62 
# port : 3306
# username : u832389661_unama1
# password : Unama2026

# Prediksi Kelembaban Tanah Gambut — AIoT Irigasi Adaptif
**3 Node ESP32 | 3 Model ML | 3 Horizon Prediksi**

---

## Urutan Eksekusi Notebook

| # | Notebook | Deskripsi | Output |
|---|---|---|---|
| 1 | `01_preprocessing_sensor.ipynb` | Load DB, cleaning data, hapus outlier IQR | `dataset/data_cleaned.csv` |
| 2 | `02_preprocessing.ipynb` | Feature engineering multi-horizon, lag, rolling, EWM | `dataset/data_train/test_{60m\|180m\|360m}.csv` |
| 3 | `03_random_forest.ipynb` | Random Forest + RandomizedSearchCV per horizon | `models/rf_model_{60m\|180m\|360m}.pkl` |
| 4 | `04_xgboost.ipynb` | XGBoost + LightGBM + RandomizedSearchCV per horizon | `models/xgb_model_*.pkl`, `lgb_model_*.pkl` |
| 5 | `05_evaluasi_model.ipynb` | Evaluasi lengkap, forecast multi-jam, rekomendasi irigasi | `dataset/hasil_rekomendasi_irigasi.csv` |

---

## Target Prediksi (Multi-Horizon)

| Horizon | Langkah | Keterangan |
|---|---|---|
| `target_60m` | 60 menit ke depan | Prediksi 1 jam |
| `target_180m` | 180 menit ke depan | Prediksi 3 jam |
| `target_360m` | 360 menit ke depan | Prediksi 6 jam |

---

## Model ML

| Model | Keunggulan |
|---|---|
| **Random Forest** | Baseline, interpretabel, robust terhadap outlier |
| **XGBoost** | Gradient boosting dioptimasi, regularisasi L1/L2/Gamma |
| **LightGBM** | Leaf-wise growth, training paling cepat, cocok data besar |

---

## Struktur Kolom Database (`sensor_readings`)

| Kolom | Tipe | Keterangan |
|---|---|---|
| `kelembaban_tanah_1/2/3` | FLOAT | **Target:** Kelembaban tanah (%) |
| `kelembaban_udara_1/2/3` | FLOAT | Kelembaban udara relatif (%) |
| `tekanan_udara_1/2/3` | FLOAT | Tekanan udara (hPa) |
| `suhu_udara_1/2/3` | FLOAT | Suhu udara (°C) |
| `curah_hujan` | FLOAT | Curah hujan (mm) |
| `waktu_pembacaan` | DATETIME | Timestamp (~1 menit interval) |
| `sensor_tanah_adc_1/2/3` | INT | **Dihapus** — nilai ADC raw tidak digunakan ML |

---

## Threshold Rekomendasi Irigasi Gambut

| Kelembaban | Status | Aksi |
|---|---|---|
| < 60% | KRITIS | Irigasi SEGERA |
| 60–70% | WASPADA | Bersiap irigasi |
| 70–80% | OPTIMAL | Tidak perlu irigasi |
| 80–90% | LEMBAB | Perhatikan drainase |
| > 90% | JENUH | Hentikan irigasi |

---

## Metrik Evaluasi

| Metrik | Keterangan |
|---|---|
| **MAE** | Mean Absolute Error (%) |
| **RMSE** | Root Mean Square Error (%) |
| **R²** | Koefisien Determinasi (0–1) |
| **MAPE** | Mean Absolute Percentage Error (%) |

---

## Instalasi Library

```bash
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn sqlalchemy pymysql joblib nbformat
```

---

## Koneksi Database

```
mysql+pymysql://u832389661_unama1:Unama2026@153.92.15.62:3306/u832389661_unama1
```
