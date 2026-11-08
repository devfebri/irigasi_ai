"""Builder notebook 04r_xgboost_lightgbm.ipynb"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md("""
# Notebook 04r — XGBoost & LightGBM Multi-Horizon (REVISI)
**Basis:** `04_xgboost.ipynb` (tidak diubah). Ruang pencarian, `n_iter=25`, dan TimeSeriesSplit 3 fold sama seperti lama (`random_state=42`).

**Apa yang direvisi dan mengapa:**
- **Perbaikan kebocoran data.** Notebook lama memakai data uji (`eval_set=[(X_test, y_test)]`) untuk early stopping, sehingga jumlah pohon dipilih dengan melihat data uji. Kini **early stopping dihilangkan**: jumlah pohon (`n_estimators`) ditentukan oleh pencarian hiperparameter (TimeSeriesSplit pada data latih), dan model final dilatih pada seluruh data latih. **Data uji tidak dipakai saat pelatihan.**
- Input/output dialihkan ke `dataset/revisi/` dan `models/revisi/` (data 2 node, fitur per segmen). File prediksi memuat `segmen`, `aktual`, dan `sekarang`.
""")

code("""
# -- CELL 1: Import -----------------------------------------------
import os, json, warnings, joblib, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import xgboost as xgb, lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform

plt.style.use('seaborn-v0_8-whitegrid')
OUT, MOUT = 'dataset/revisi', 'models/revisi'
os.makedirs(OUT, exist_ok=True); os.makedirs(MOUT, exist_ok=True)
HORIZONS = ['60m', '180m', '360m']
TARGET_MAP = {h: f'target_{h}' for h in HORIZONS}
print('XGBoost', xgb.__version__, '| LightGBM', lgb.__version__)

def evaluasi(y_true, y_pred, label=''):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred); rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred); mape = float(np.mean(np.abs((y_true - y_pred)/(np.abs(y_true)+1e-8)))*100)
    if label: print(f'  [{label}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  MAPE={mape:.2f}%')
    return {'MAE': float(mae), 'RMSE': rmse, 'R2': float(r2), 'MAPE': mape}

def py(v):  # numpy -> python (agar bisa di-JSON)
    return v.item() if hasattr(v, 'item') else v

# -- Ruang pencarian (sama seperti lama) ----------------------------
xgb_param_dist = {
    'n_estimators': randint(200, 601), 'max_depth': randint(4, 11), 'learning_rate': uniform(0.01, 0.14),
    'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.6, 0.4), 'min_child_weight': randint(3, 11),
    'gamma': uniform(0, 0.5), 'reg_alpha': uniform(0, 0.5), 'reg_lambda': uniform(0.5, 2.0)}
lgb_param_dist = {
    'n_estimators': randint(200, 601), 'max_depth': randint(4, 11), 'learning_rate': uniform(0.01, 0.14),
    'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.6, 0.4), 'min_child_samples': randint(10, 51),
    'reg_alpha': uniform(0, 0.5), 'reg_lambda': uniform(0.5, 2.0), 'num_leaves': randint(20, 80)}
N_ITER, N_CV = 25, 3

def muat(horizon):
    target = TARGET_MAP[horizon]
    tr = pd.read_csv(f'{OUT}/data_train_{horizon}.csv', parse_dates=['waktu_pembacaan'])
    te = pd.read_csv(f'{OUT}/data_test_{horizon}.csv',  parse_dates=['waktu_pembacaan'])
    with open(f'{MOUT}/feature_cols_{horizon}.json') as f: fc = json.load(f)
    return tr, te, fc, target

def simpan_prediksi(te, target, pred, nama, horizon):
    pd.DataFrame({'waktu_pembacaan': te['waktu_pembacaan'], 'segmen': te['segmen'], 'aktual': te[target].values,
                  'sekarang': te['kelembaban_tanah_avg'].values, f'prediksi_{nama}': pred}
                 ).to_csv(f'{OUT}/prediksi_{nama}_{horizon}.csv', index=False)
""")

code("""
# -- CELL 4: XGBoost (tanpa early stopping) -------
xgb_results = {}
for horizon in HORIZONS:
    t0 = time.time()
    tr, te, fc, target = muat(horizon)
    X_tr, y_tr = tr[fc].values.astype(np.float32), tr[target].values.astype(np.float32)
    X_te, y_te = te[fc].values.astype(np.float32), te[target].values.astype(np.float32)
    print(f'\\n=== XGBoost {horizon} | train={len(X_tr):,} test={len(X_te):,} ===')

    search = RandomizedSearchCV(
        XGBRegressor(tree_method='hist', random_state=42, n_jobs=-1, eval_metric='rmse', verbosity=0),
        xgb_param_dist, n_iter=N_ITER, cv=TimeSeriesSplit(n_splits=N_CV),
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_tr, y_tr)                                # CV hanya memakai data latih
    best = {a: py(b) for a, b in search.best_params_.items()}
    cv_rmse = float(-search.best_score_)
    print(f'  CV-RMSE {cv_rmse:.4f} | {best}')

    model = XGBRegressor(**best, tree_method='hist', random_state=42, n_jobs=-1, eval_metric='rmse', verbosity=0)
    model.fit(X_tr, y_tr, verbose=False)                  # seluruh data latih, tanpa early stopping
    print(f'  n_estimators={best["n_estimators"]}')

    p_tr, p_te = model.predict(X_tr), model.predict(X_te)
    m_tr = evaluasi(y_tr, p_tr, f'XGB train-{horizon}'); m_te = evaluasi(y_te, p_te, f'XGB test -{horizon}')
    path = f'{MOUT}/xgb_model_{horizon}.pkl'
    joblib.dump(model, path, compress=3); model.save_model(f'{MOUT}/xgb_model_{horizon}.json')
    simpan_prediksi(te, target, p_te, 'xgb', horizon)
    xgb_results[horizon] = {'best_params': best, 'cv_rmse': cv_rmse, 'n_estimators': int(best['n_estimators']),
        'n_train': int(len(X_tr)), 'train_metrics': m_tr, 'test_metrics': m_te,
        'model_size_mb': float(os.path.getsize(path)/1e6), 'waktu_latih_detik': float(time.time()-t0)}
    print(f'  selesai {time.time()-t0:.0f} detik')
with open(f'{MOUT}/xgb_metrics_all.json', 'w') as f: json.dump(xgb_results, f, indent=2)
""")

code("""
# -- CELL 5: LightGBM (tanpa early stopping) ------
lgb_results = {}
for horizon in HORIZONS:
    t0 = time.time()
    tr, te, fc, target = muat(horizon)
    X_tr, y_tr = tr[fc].values.astype(np.float32), tr[target].values.astype(np.float32)
    X_te, y_te = te[fc].values.astype(np.float32), te[target].values.astype(np.float32)
    print(f'\\n=== LightGBM {horizon} | train={len(X_tr):,} test={len(X_te):,} ===')

    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
        lgb_param_dist, n_iter=N_ITER, cv=TimeSeriesSplit(n_splits=N_CV),
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_tr, y_tr)
    best = {a: py(b) for a, b in search.best_params_.items()}
    cv_rmse = float(-search.best_score_)
    print(f'  CV-RMSE {cv_rmse:.4f} | {best}')

    model = LGBMRegressor(**best, random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_tr, y_tr)                                 # seluruh data latih, tanpa early stopping
    print(f'  n_estimators={best["n_estimators"]}')

    p_tr, p_te = model.predict(X_tr), model.predict(X_te)
    m_tr = evaluasi(y_tr, p_tr, f'LGB train-{horizon}'); m_te = evaluasi(y_te, p_te, f'LGB test -{horizon}')
    path = f'{MOUT}/lgb_model_{horizon}.pkl'
    joblib.dump(model, path, compress=3)
    simpan_prediksi(te, target, p_te, 'lgb', horizon)
    lgb_results[horizon] = {'best_params': best, 'cv_rmse': cv_rmse, 'n_estimators': int(best['n_estimators']),
        'n_train': int(len(X_tr)), 'train_metrics': m_tr, 'test_metrics': m_te,
        'model_size_mb': float(os.path.getsize(path)/1e6), 'waktu_latih_detik': float(time.time()-t0)}
    print(f'  selesai {time.time()-t0:.0f} detik')
with open(f'{MOUT}/lgb_metrics_all.json', 'w') as f: json.dump(lgb_results, f, indent=2)
""")

code("""
# -- CELL 6-7: Ringkasan & plot ------------------------------------
with open(f'{MOUT}/rf_metrics_all.json') as f: rf_results = json.load(f)
print(f'{"Horizon":<8}{"Model":<13}{"MAE":>8}{"RMSE":>8}{"R2":>8}{"MAPE":>8}')
for h in HORIZONS:
    for nama, res, key in [('RF', rf_results, 'test_metrics'), ('XGBoost', xgb_results, 'test_metrics'),
                           ('LightGBM', lgb_results, 'test_metrics'), ('Persistensi', rf_results, 'persistensi_metrics')]:
        m = res[h][key]; print(f'{h:<8}{nama:<13}{m["MAE"]:>8.3f}{m["RMSE"]:>8.3f}{m["R2"]:>8.3f}{m["MAPE"]:>8.2f}')
    print()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, h in zip(axes, HORIZONS):
    v = [xgb_results[h]['test_metrics']['RMSE'], lgb_results[h]['test_metrics']['RMSE']]
    bars = ax.bar(['XGBoost', 'LightGBM'], v, color=['#43A047', '#FF6F00'], width=0.5)
    for b, x in zip(bars, v): ax.text(b.get_x()+b.get_width()/2, x*1.01, f'{x:.3f}', ha='center', fontweight='bold')
    ax.set_title(f'Horizon {h}'); ax.set_ylabel('RMSE (%)')
plt.tight_layout(); plt.savefig(f'{OUT}/plot_xgb_lgb_rmse_revisi.png', dpi=150); plt.show()
print('Lanjut ke 05r_evaluasi_model.ipynb')
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '04r_xgboost_lightgbm.ipynb')
print('ditulis')
