"""Builder notebook 02r_feature_engineering.ipynb"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md("""
# Notebook 02r — Feature Engineering per Segmen (REVISI)
**Basis:** `02_preprocessing.ipynb` (tidak diubah). Input `dataset/revisi/data_cleaned_revisi.csv`, output `dataset/revisi/` dan `models/revisi/`.

**Apa yang direvisi dan mengapa:**
- **E1 – tidak melintasi celah.** Lag, rolling, diff, EWM dihitung **per segmen** (`groupby('segmen')`). Target tidak lagi memakai `shift(-steps)`, melainkan nilai pada waktu t+h yang dicari dengan `merge_asof` (toleransi 2 menit) dan dibuang (NaN) bila baris target berada di segmen lain.
- **E2 – Node 3 dikeluarkan.** Target = rata-rata Node 1 dan Node 2; fitur Node 3 dihapus (jumlah fitur = 73). Versi 3 node dibuat terpisah di `dataset/revisi/sens3/` hanya untuk uji sensitivitas di 05r.
- `waktu_pembacaan` dan `segmen` bukan fitur. Split tetap kronologis 80/20.
- Ditambahkan sel verifikasi anti-kebocoran lintas segmen.
""")

code("""
# -- CELL 1: Import & Load ---------------------------------------
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

OUT = 'dataset/revisi'; MOUT = 'models/revisi'
os.makedirs(OUT, exist_ok=True); os.makedirs(MOUT, exist_ok=True); os.makedirs(f'{OUT}/sens3', exist_ok=True)
HORIZONS = {'60m': 60, '180m': 180, '360m': 360}
META_COLS = ['waktu_pembacaan', 'segmen']
TARGET_COLS = [f'target_{h}' for h in HORIZONS]

df0 = pd.read_csv(f'{OUT}/data_cleaned_revisi.csv', parse_dates=['waktu_pembacaan'])
df0 = df0.sort_values('waktu_pembacaan').reset_index(drop=True)
print(f'Baris {len(df0):,} | segmen {df0["segmen"].nunique()} | {df0["waktu_pembacaan"].min()} -> {df0["waktu_pembacaan"].max()}')
""")

code("""
# -- CELL 2: Target (merge_asof, per segmen) ----------------------
def buat_target(df, nodes):
    kt = [f'kelembaban_tanah_{n}' for n in nodes]
    df['kelembaban_tanah_avg'] = df[kt].mean(axis=1)
    right = df[['waktu_pembacaan', 'kelembaban_tanah_avg', 'segmen']].rename(
        columns={'kelembaban_tanah_avg': '_val', 'segmen': '_seg'})
    for name, menit in HORIZONS.items():
        left = pd.DataFrame({'waktu_target': df['waktu_pembacaan'] + pd.Timedelta(minutes=menit),
                             'segmen': df['segmen']})
        m = pd.merge_asof(left, right, left_on='waktu_target', right_on='waktu_pembacaan',
                          direction='nearest', tolerance=pd.Timedelta('2min'))
        val = m['_val'].where(m['_seg'] == m['segmen'])      # beda segmen -> NaN
        df[f'target_{name}'] = val.values
        print(f'  target_{name}: valid {df[f"target_{name}"].notna().sum():,} / {len(df):,}')
    return df

df = buat_target(df0.copy(), [1, 2])
""")

code("""
# -- CELL 3: Fitur waktu (sama seperti lama) ----------------------
def buat_waktu(df):
    w = df['waktu_pembacaan']
    df['jam'] = w.dt.hour; df['menit'] = w.dt.minute; df['hari'] = w.dt.day
    df['bulan'] = w.dt.month; df['hari_minggu'] = w.dt.dayofweek
    df['is_weekend'] = (df['hari_minggu'] >= 5).astype(int)
    df['jam_sin'] = np.sin(2*np.pi*df['jam']/24); df['jam_cos'] = np.cos(2*np.pi*df['jam']/24)
    df['bulan_sin'] = np.sin(2*np.pi*df['bulan']/12); df['bulan_cos'] = np.cos(2*np.pi*df['bulan']/12)
    df['periode_hari'] = pd.cut(df['jam'], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)
    return df

df = buat_waktu(df)
""")

code("""
# -- CELL 4: Lag per segmen ---------------------------------------
def buat_lag(df, nodes):
    g = df.groupby('segmen')
    for lag in [1, 5, 10, 30, 60, 120, 180, 360]:
        df[f'kt_avg_lag_{lag}'] = g['kelembaban_tanah_avg'].shift(lag)
    for n in nodes:
        for lag in [1, 5, 60]:
            df[f'kt{n}_lag_{lag}'] = g[f'kelembaban_tanah_{n}'].shift(lag)
    for feat, safe in [('kelembaban_udara_1', 'kelembaban_udara'), ('suhu_udara_1', 'suhu_udara'), ('curah_hujan', 'curah_hujan')]:
        for lag in [1, 30, 60]:
            df[f'{safe}_lag_{lag}'] = g[feat].shift(lag)
    return df

df = buat_lag(df, [1, 2])
""")

code("""
# -- CELL 5: Rolling & diff per segmen ----------------------------
def buat_rolling(df):
    seg = df['segmen']
    s = df.groupby('segmen')['kelembaban_tanah_avg'].shift(1)
    sh = df.groupby('segmen')['curah_hujan'].shift(1)
    for wname, w in {'1jam': 60, '3jam': 180, '6jam': 360}.items():
        r = s.groupby(seg).rolling(w, min_periods=10)
        for st in ['mean', 'std', 'min', 'max']:
            df[f'kt_roll_{st}_{wname}'] = getattr(r, st)().reset_index(level=0, drop=True)
        df[f'hujan_roll_sum_{wname}'] = (sh.groupby(seg).rolling(w, min_periods=1).sum()
                                         .reset_index(level=0, drop=True))
    for d in [1, 5, 30, 60]:
        df[f'kt_avg_diff_{d}'] = df.groupby('segmen')['kelembaban_tanah_avg'].diff(d)
    return df

df = buat_rolling(df)
""")

code("""
# -- CELL 6: EWM & interaksi --------------------------------------
def buat_ewm_interaksi(df, nodes):
    g = df.groupby('segmen')
    for span in [12, 60, 180]:
        df[f'kt_ewm_{span}'] = g['kelembaban_tanah_avg'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['ku_avg'] = df[[f'kelembaban_udara_{n}' for n in nodes]].mean(axis=1)
    df['suhu_avg'] = df[[f'suhu_udara_{n}' for n in nodes]].mean(axis=1)
    df['tekanan_avg'] = df[[f'tekanan_udara_{n}' for n in nodes]].mean(axis=1)
    for n in nodes:
        ku, su = df[f'kelembaban_udara_{n}'], df[f'suhu_udara_{n}']
        df[f'vpd_{n}'] = (1 - ku/100) * 0.6108 * np.exp(17.27*su/(su + 237.3))
    df['vpd_avg'] = df[[f'vpd_{n}' for n in nodes]].mean(axis=1)
    df['kt_diff_12'] = df['kelembaban_tanah_1'] - df['kelembaban_tanah_2']
    if 3 in nodes:                                   # hanya versi sensitivitas
        df['kt_diff_13'] = df['kelembaban_tanah_1'] - df['kelembaban_tanah_3']
        df['kt_diff_23'] = df['kelembaban_tanah_2'] - df['kelembaban_tanah_3']
    return df

df = buat_ewm_interaksi(df, [1, 2])
""")

code("""
# -- CELL 7: Finalisasi, daftar fitur, split, simpan ---------------
def finalisasi(df, folder, mfolder, tag=''):
    feat_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    hasil = {}
    for h in HORIZONS:
        t = f'target_{h}'
        d = df[META_COLS + feat_cols + [t]].dropna()
        k = int(len(d)*0.8)
        tr, te = d.iloc[:k], d.iloc[k:]
        tr.to_csv(f'{folder}/data_train_{h}.csv', index=False)
        te.to_csv(f'{folder}/data_test_{h}.csv', index=False)
        with open(f'{mfolder}/feature_cols_{h}.json', 'w') as f: json.dump(feat_cols, f, indent=2)
        hasil[h] = {'n_total': len(d), 'n_train': len(tr), 'n_test': len(te),
                    'train_awal': str(tr['waktu_pembacaan'].min()), 'train_akhir': str(tr['waktu_pembacaan'].max()),
                    'test_awal': str(te['waktu_pembacaan'].min()), 'test_akhir': str(te['waktu_pembacaan'].max()),
                    'segmen_test': sorted(te['segmen'].unique().tolist())}
        print(f'{tag}{h}: total {len(d):,} | train {len(tr):,} | test {len(te):,} | test {te["waktu_pembacaan"].min()} -> {te["waktu_pembacaan"].max()}')
    return feat_cols, hasil

feat_cols, info2 = finalisasi(df, OUT, MOUT)

kelompok = {
 'mentah Node 1-2 + hujan': [c for c in feat_cols if c in ['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_udara_1','kelembaban_udara_2','suhu_udara_1','suhu_udara_2','tekanan_udara_1','tekanan_udara_2','curah_hujan']],
 'waktu': ['jam','menit','hari','bulan','hari_minggu','is_weekend','jam_sin','jam_cos','bulan_sin','bulan_cos','periode_hari'],
 'target saat ini (kt_avg)': ['kelembaban_tanah_avg'],
 'lag kt_avg': [c for c in feat_cols if c.startswith('kt_avg_lag_')],
 'lag per node': [c for c in feat_cols if c.startswith(('kt1_lag_','kt2_lag_'))],
 'lag udara/suhu/hujan': [c for c in feat_cols if c.startswith(('kelembaban_udara_lag_','suhu_udara_lag_','curah_hujan_lag_'))],
 'rolling kt': [c for c in feat_cols if c.startswith('kt_roll_')],
 'rolling hujan': [c for c in feat_cols if c.startswith('hujan_roll_')],
 'diff kt_avg': [c for c in feat_cols if c.startswith('kt_avg_diff_')],
 'EWM': [c for c in feat_cols if c.startswith('kt_ewm_')],
 'rata-rata node': ['ku_avg','suhu_avg','tekanan_avg'],
 'VPD': ['vpd_1','vpd_2','vpd_avg'],
 'selisih node': ['kt_diff_12'],
}
print('\\n=== Fitur per kelompok ===')
for k, v in kelompok.items(): print(f'{k:<26}{len(v):>3}  {v}')
terpakai = sum(kelompok.values(), [])
assert sorted(terpakai) == sorted(feat_cols), set(feat_cols) ^ set(terpakai)
print(f'\\nTOTAL FITUR = {len(feat_cols)}')
assert len(feat_cols) == 73, 'jumlah fitur harus 73'
""")

code("""
# -- CELL 8: Verifikasi tidak ada kebocoran lintas segmen ----------
rng = np.random.default_rng(42)
seg = df['segmen']

# (a) target: sampel acak -> selisih waktu ~h menit & segmen sama & nilai cocok
for h, menit in HORIZONS.items():
    d = df.dropna(subset=[f'target_{h}'])
    idx = rng.choice(d.index.values, size=min(2000, len(d)), replace=False)
    selisih, ok_seg, ok_val = [], True, True
    wk = df['waktu_pembacaan'].values
    for i in idx:
        tt = wk[i] + np.timedelta64(menit, 'm')
        j = np.abs(wk - tt).argmin()
        selisih.append((wk[j] - wk[i]) / np.timedelta64(1, 'm'))
        ok_seg &= bool(seg[j] == seg[i])
        ok_val &= bool(np.isclose(df.at[j, 'kelembaban_tanah_avg'], df.at[i, f'target_{h}']))
    selisih = np.array(selisih)
    print(f'target_{h}: n={len(idx)} | selisih waktu {selisih.min():.2f}..{selisih.max():.2f} menit (harus {menit}+-2) | segmen sama={ok_seg} | nilai cocok={ok_val}')
    assert abs(selisih - menit).max() <= 2 and ok_seg and ok_val

# (b) lag/diff: fitur bernilai hanya jika baris sumber ada di segmen yang sama
for lag in [1, 5, 10, 30, 60, 120, 180, 360]:
    ada = df[f'kt_avg_lag_{lag}'].notna()
    assert (seg.shift(lag)[ada] == seg[ada]).all(), f'lag {lag} melintasi segmen'
for d_ in [1, 5, 30, 60]:
    ada = df[f'kt_avg_diff_{d_}'].notna()
    assert (seg.shift(d_)[ada] == seg[ada]).all(), f'diff {d_} melintasi segmen'
print('Lag & diff: tidak ada yang mengambil baris dari segmen lain')

# (c) baris pertama tiap segmen: semua lag NaN dan rolling NaN
awal = df.groupby('segmen').head(1).index
for c in ['kt_avg_lag_1', 'kt_roll_mean_1jam', 'kt_ewm_12', 'kt_avg_diff_1']:
    assert df.loc[awal, c].isna().all(), c
print('Baris awal tiap segmen: lag/rolling/EWM/diff = NaN (tidak meminjam segmen sebelumnya)')

# (d) rolling 6 jam: min_periods=10 -> NaN pada 10 baris pertama tiap segmen
rn = df.groupby('segmen').cumcount()
assert df.loc[rn < 10, 'kt_roll_mean_6jam'].isna().all()
print('Rolling: NaN pada <10 baris pertama tiap segmen (sesuai min_periods)')
""")

code("""
# -- CELL 9: Versi 3 node (khusus uji sensitivitas, dipakai di 05r) -
raw3 = pd.read_csv(f'{OUT}/data_node3_qc.csv', parse_dates=['waktu_pembacaan'])
kolom3 = ['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3',
          'kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3',
          'suhu_udara_1','suhu_udara_2','suhu_udara_3',
          'tekanan_udara_1','tekanan_udara_2','tekanan_udara_3','curah_hujan']
n_awal3 = len(raw3)
d3 = raw3.dropna(subset=kolom3).sort_values('waktu_pembacaan').reset_index(drop=True)
d3['segmen'] = (d3['waktu_pembacaan'].diff() > pd.Timedelta(minutes=10)).cumsum().astype(int)
print(f'sens3: {n_awal3:,} -> {len(d3):,} baris valid | {d3["segmen"].nunique()} segmen')
d3.to_csv(f'{OUT}/sens3/data_cleaned_sens3.csv', index=False)

d3 = buat_target(d3, [1, 2, 3])
d3 = buat_waktu(d3)
d3 = buat_lag(d3, [1, 2, 3])
d3 = buat_rolling(d3)
d3 = buat_ewm_interaksi(d3, [1, 2, 3])
os.makedirs(f'{OUT}/sens3', exist_ok=True)
feat3, info3 = finalisasi(d3, f'{OUT}/sens3', f'{OUT}/sens3', tag='sens3 ')
print(f'Jumlah fitur sens3: {len(feat3)}')
""")

code("""
# -- CELL 10: Simpan ringkasan ------------------------------------
ringkas = {'jumlah_fitur': len(feat_cols), 'kelompok_fitur': {k: len(v) for k, v in kelompok.items()},
           'split_2node': info2, 'jumlah_fitur_sens3': len(feat3), 'split_sens3': info3,
           'baris_sens3_valid': int(len(d3)), 'segmen_sens3': int(d3['segmen'].nunique())}
with open(f'{OUT}/feature_report_revisi.json', 'w') as f: json.dump(ringkas, f, indent=2)
print('[OK] feature_report_revisi.json | Lanjut ke 03r_random_forest.ipynb')
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '02r_feature_engineering.ipynb')
print('ditulis')
