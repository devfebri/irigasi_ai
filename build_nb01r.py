"""Builder notebook 01r_preprocessing_sensor.ipynb.
Pemakaian: python build_nb01r.py [ask|keep|drop]
  ask  : notebook berhenti setelah menghitung outlier 6i (menunggu keputusan)
  keep : outlier dipertahankan (diinterpolasi, run <=10 baris)
  drop : langkah 6i dihapus
"""
import sys
import nbformat as nbf

MODE = sys.argv[1] if len(sys.argv) > 1 else 'ask'
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md("""
# Notebook 01r — Preprocessing Sensor (REVISI)
**Basis:** `01_preprocessing_sensor.ipynb` (tidak diubah). Output ke `dataset/revisi/`.

**Apa yang direvisi dan mengapa (komentar editor E1–E2):**
- **E1 – celah waktu.** Interpolasi linear untuk kekosongan panjang (hingga 72 jam) diganti interpolasi hanya untuk kekosongan <=10 baris (`limit_area='inside'`, tanpa `bfill/ffill`). Kekosongan lebih panjang dibiarkan NaN, barisnya dibuang, dan data dipecah menjadi **segmen** (selisih waktu >10 menit = segmen baru) agar model tidak mempelajari garis lurus buatan.
- **E2 – Node 3 anomali.** Node 3 dikeluarkan dari target dan fitur. Datanya disimpan terpisah (`data_node3_qc.csv`) hanya untuk uji sensitivitas, disertai sel QC.
- Bacaan 0 pada suhu dan kelembaban udara kini dianggap gagal baca (NaN).
- Bacaan kelembaban tanah < 10% (termasuk 0) dianggap gagal baca (NaN), karena hanya muncul di tepi celah data saat sensor baru menyala atau koneksi putus.
- Data dibatasi sampai `waktu_pembacaan <= 2026-10-19 08:50:06` (periode naskah).
- Kredensial DB dibaca dari `.env`. Database hanya dibaca (SELECT).
""")

code("""
# -- CELL 1: Import Library --------------------------------------
import os, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

OUT = 'dataset/revisi'
os.makedirs(OUT, exist_ok=True)
BATAS_WAKTU = '2026-10-19 08:50:06'
MAX_RUN     = 10        # kekosongan maksimum (baris) yang boleh diinterpolasi
GAP_MENIT   = 10        # selisih waktu > ini => segmen baru
print('[OK] Library diimport | pandas', pd.__version__)
""")

code("""
# -- CELL 2: Koneksi DB (dari .env) & Load Data (SELECT saja) ----
load_dotenv()
url = (f"mysql+pymysql://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
       f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}")
engine = create_engine(url)

query = text("SELECT * FROM sensor_readings WHERE waktu_pembacaan <= :batas "
             "ORDER BY waktu_pembacaan ASC")
with engine.connect() as conn:
    df_raw = pd.read_sql(query, conn, params={'batas': BATAS_WAKTU})

print('=== Informasi Data Mentah ===')
print(f'Jumlah baris : {df_raw.shape[0]:,}')
print(f'Kolom        : {list(df_raw.columns)}')
print(f'Periode      : {df_raw["waktu_pembacaan"].min()} -> {df_raw["waktu_pembacaan"].max()}')
print(df_raw.head(3))
""")

code("""
# -- CELL 3: Analisis Missing Values -----------------------------
miss = df_raw.isnull().sum()
print(pd.DataFrame({'Missing': miss, 'Persen': (miss/len(df_raw)*100).round(3)})
      .query('Missing > 0') if miss.any() else '[OK] Tidak ada missing value')
""")

code("""
# -- CELL 4: Analisis Nilai Nol per Kolom ------------------------
sensor_cols = [c for c in [
    'kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3',
    'kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3',
    'suhu_udara_1','suhu_udara_2','suhu_udara_3',
    'tekanan_udara_1','tekanan_udara_2','tekanan_udara_3','curah_hujan'] if c in df_raw.columns]
z = (df_raw[sensor_cols] == 0).sum()
print(pd.DataFrame({'Jumlah_Nol': z, 'Persen(%)': (z/len(df_raw)*100).round(3)}))
""")

code("""
# -- CELL 5: Distribusi Data Mentah ------------------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Distribusi Sensor Data Mentah per Node', fontsize=14, fontweight='bold')
groups = [(['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3'], 'Kelembaban Tanah (%)'),
          (['kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3'], 'Kelembaban Udara (%)'),
          (['suhu_udara_1','suhu_udara_2','suhu_udara_3'], 'Suhu Udara (C)')]
for r, (cols, yl) in enumerate(groups):
    for ci, col in enumerate(cols):
        axes[r][ci].hist(df_raw[col].dropna(), bins=50, alpha=0.7, edgecolor='white')
        axes[r][ci].set_title(f'Node {ci+1}'); axes[r][ci].set_xlabel(yl)
plt.tight_layout()
plt.savefig(f'{OUT}/plot_distribusi_raw_revisi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""
# -- CELL 6a-6e: langkah awal (sama seperti notebook lama) --------
LAPORAN = {'baris_awal': int(len(df_raw)), 'batas_waktu_data': BATAS_WAKTU}
df = df_raw.copy()

# 6a. waktu
df['waktu_pembacaan'] = pd.to_datetime(df['waktu_pembacaan'], errors='coerce')
LAPORAN['dibuang_timestamp_invalid'] = int(df['waktu_pembacaan'].isna().sum())
df = df.dropna(subset=['waktu_pembacaan']).sort_values('waktu_pembacaan').reset_index(drop=True)

# 6b. duplikat timestamp
LAPORAN['dibuang_duplikat'] = int(df.duplicated(subset=['waktu_pembacaan']).sum())
df = df.drop_duplicates(subset=['waktu_pembacaan'], keep='first').reset_index(drop=True)

# 6c. kolom tak dipakai
drop_cols = ['id','backup_status','created_at','updated_at',
             'sensor_tanah_adc_1','sensor_tanah_adc_2','sensor_tanah_adc_3']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 6d. numerik
num_cols = [c for c in df.columns if c != 'waktu_pembacaan']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# 6e. ketiga sensor tanah = 0
kt_cols = ['kelembaban_tanah_1','kelembaban_tanah_2','kelembaban_tanah_3']
m = (df[kt_cols] == 0).all(axis=1)
LAPORAN['dibuang_semua_sensor_tanah_nol'] = int(m.sum())
df = df[~m].reset_index(drop=True)

# Lonjakan pada data setelah 6e (nol belum ditangani) untuk QC
df_pasca6e = df[['waktu_pembacaan'] + kt_cols].copy()

for k in ['dibuang_timestamp_invalid','dibuang_duplikat','dibuang_semua_sensor_tanah_nol']:
    print(f'{k}: {LAPORAN[k]:,}')
print(f'Baris setelah 6a-6e: {len(df):,}')

# Segmen awal (seg0) HANYA untuk interpolasi: agar nilai tidak diinterpolasi melintasi celah waktu
df['seg0'] = (df['waktu_pembacaan'].diff() > pd.Timedelta(minutes=GAP_MENIT)).cumsum()

def interp_pendek(s, maxrun=MAX_RUN):
    \"\"\"Interpolasi linear hanya untuk run NaN <= maxrun baris (limit_area='inside').
    Run NaN lebih panjang dibiarkan NaN seluruhnya (bukan terisi sebagian).\"\"\"
    isn = s.isna()
    rid = (isn != isn.shift()).cumsum()
    runlen = isn.groupby(rid).transform('sum')
    filled = s.interpolate(method='linear', limit_area='inside')
    return filled.where(~(isn & (runlen > maxrun)), np.nan)

def bersihkan(col, mask_invalid):
    \"\"\"Tandai invalid -> NaN, interpolasi pendek per seg0. Kembalikan (n_invalid, n_tersisa_NaN).\"\"\"
    n = int(mask_invalid.sum())
    df.loc[mask_invalid, col] = np.nan
    df[col] = df.groupby('seg0')[col].transform(interp_pendek)
    return n, int(df[col].isna().sum())
""")

code("""
# -- CELL 6f-6h: nol/invalid -> NaN, interpolasi pendek -----------
rin = {}
# 6f. kelembaban tanah < 10% dianggap gagal baca (termasuk 0).
#     Tanah gambut di lokasi tidak pernah sekering ini; bacaan < 10% muncul
#     hanya di tepi celah data (sensor baru menyala / koneksi putus).
BATAS_MIN_KT = 10
for c in kt_cols:
    rin[c] = bersihkan(c, df[c] < BATAS_MIN_KT)
LAPORAN['batas_min_kelembaban_tanah'] = BATAS_MIN_KT
# 6g. tekanan di luar 800-1100 hPa
for c in ['tekanan_udara_1','tekanan_udara_2','tekanan_udara_3']:
    rin[c] = bersihkan(c, (df[c] < 800) | (df[c] > 1100))
# 6h. kelembaban udara >100 atau 0 ; suhu udara 0 (BARU)
for c in ['kelembaban_udara_1','kelembaban_udara_2','kelembaban_udara_3']:
    rin[c] = bersihkan(c, (df[c] > 100) | (df[c] == 0))
for c in ['suhu_udara_1','suhu_udara_2','suhu_udara_3']:
    rin[c] = bersihkan(c, df[c] == 0)

print(f'{"kolom":<22}{"invalid":>9}{"NaN tersisa":>13}')
for c, (n, left) in rin.items():
    print(f'{c:<22}{n:>9,}{left:>13,}')
LAPORAN['invalid_dan_nan_tersisa_per_kolom'] = {c: {'invalid': a, 'nan_tersisa': b} for c, (a, b) in rin.items()}
""")

code("""
# -- CELL 6i: Outlier (aturan LAMA) -- hitung dulu, JANGAN ubah ---
cnt_out = {}
for c in kt_cols:
    s = df[c].dropna()
    q1, q3 = s.quantile(0.01), s.quantile(0.99)
    iqr = q3 - q1
    low, high = max(q1 - 3*iqr, 0.0), min(q3 + 3*iqr, 100.0)
    mask = (df[c] < low) | (df[c] > high)
    cnt_out[c] = {'batas_bawah': float(low), 'batas_atas': float(high), 'jumlah_outlier': int(mask.sum())}
    print(f'{c}: batas [{low:.1f}, {high:.1f}] -> {int(mask.sum()):,} nilai kena aturan outlier lama')
LAPORAN['outlier_aturan_lama'] = cnt_out
""")

if MODE == 'ask':
    pass
else:
    code(f"""
# -- CELL 6i (lanjutan): keputusan pengguna -----------------------
OUTLIER_MODE = '{MODE}'   # 'keep' = nilai outlier -> NaN lalu interpolasi pendek ; 'drop' = langkah dihapus
LAPORAN['outlier_mode'] = OUTLIER_MODE
if OUTLIER_MODE == 'keep':
    for c in kt_cols:
        n, _ = bersihkan(c, (df[c] < cnt_out[c]['batas_bawah']) | (df[c] > cnt_out[c]['batas_atas']))
        print(f'{{c}}: {{n:,}} outlier -> NaN -> interpolasi pendek')
else:
    print('Langkah 6i (outlier) dihapus sesuai keputusan.')
""")

    code("""
# -- CELL 6j: pemilihan node (Node 1 & 2 + curah hujan) ----------
MODEL_COLS = ['kelembaban_tanah_1','kelembaban_tanah_2',
              'kelembaban_udara_1','kelembaban_udara_2',
              'suhu_udara_1','suhu_udara_2',
              'tekanan_udara_1','tekanan_udara_2','curah_hujan']
NODE3_COLS = ['kelembaban_tanah_3','kelembaban_udara_3','suhu_udara_3','tekanan_udara_3']
df_all = df[['waktu_pembacaan'] + MODEL_COLS + NODE3_COLS].copy()
df_all.to_csv(f'{OUT}/data_node3_qc.csv', index=False)   # semua node (NaN masih ada) -> sensitivitas Node 3
print(f'[OK] {OUT}/data_node3_qc.csv ({len(df_all):,} baris, semua node, setelah pembersihan)')

# -- CELL 6k: buang baris tidak valid ----------------------------
n0 = len(df_all)
na = df_all[MODEL_COLS].isna()
dibuang_kt1 = int(na['kelembaban_tanah_1'].sum())
dibuang_kt2 = int(na['kelembaban_tanah_2'].sum())
dibuang_lain = int((na.drop(columns=['kelembaban_tanah_1','kelembaban_tanah_2']).any(axis=1)
                    & ~na[['kelembaban_tanah_1','kelembaban_tanah_2']].any(axis=1)).sum())
dv = df_all.dropna(subset=MODEL_COLS).copy()
LAPORAN['baris_dibuang_kekosongan_panjang'] = {
    'node1_kt_nan': dibuang_kt1, 'node2_kt_nan': dibuang_kt2,
    'hanya_kolom_lain_nan': dibuang_lain, 'total_dibuang': int(n0 - len(dv))}
print(f'Baris dibuang: total {n0-len(dv):,} | NaN kt1={dibuang_kt1:,} kt2={dibuang_kt2:,} | hanya kolom lain={dibuang_lain:,}')

# -- CELL 6l: segmentasi -----------------------------------------
dv = dv.sort_values('waktu_pembacaan').reset_index(drop=True)
dv['segmen'] = ((dv['waktu_pembacaan'].diff() > pd.Timedelta(minutes=GAP_MENIT)).cumsum()).astype(int)
dv['kelembaban_tanah_avg'] = dv[['kelembaban_tanah_1','kelembaban_tanah_2']].mean(axis=1)

seg = dv.groupby('segmen')['waktu_pembacaan'].agg(['min','max','count'])
seg['durasi_jam'] = (seg['max'] - seg['min']).dt.total_seconds()/3600
seg24 = seg[seg['durasi_jam'] >= 24]
gap_terpanjang = dv['waktu_pembacaan'].diff().max()
ig = dv['waktu_pembacaan'].diff().idxmax()

print(f'Baris awal            : {LAPORAN["baris_awal"]:,}')
print(f'Baris valid           : {len(dv):,}')
print(f'Jumlah segmen         : {len(seg)}')
print(f'Segmen >= 24 jam      : {len(seg24)}  ({seg24["count"].sum()/len(dv)*100:.1f}% data)')
print(f'Celah terpanjang      : {gap_terpanjang}  ({dv.loc[ig-1,"waktu_pembacaan"]} -> {dv.loc[ig,"waktu_pembacaan"]})')
sebelum, sesudah = dv.iloc[:ig], dv.iloc[ig:]
deret = [{'nama': 'sebelum celah terpanjang', 'awal': str(sebelum['waktu_pembacaan'].min()), 'akhir': str(sebelum['waktu_pembacaan'].max()), 'baris': int(len(sebelum))},
         {'nama': 'sesudah celah terpanjang', 'awal': str(sesudah['waktu_pembacaan'].min()), 'akhir': str(sesudah['waktu_pembacaan'].max()), 'baris': int(len(sesudah))}]
print('Dua deret utama (dipisah oleh celah terpanjang):')
for d in deret:
    print(f"   {d['nama']}: {d['awal']} -> {d['akhir']}  ({d['baris']:,} baris)")

LAPORAN.update({
    'baris_valid': int(len(dv)), 'jumlah_segmen': int(len(seg)),
    'jumlah_segmen_ge_24jam': int(len(seg24)),
    'persen_data_di_segmen_ge_24jam': float(seg24['count'].sum()/len(dv)*100),
    'celah_terpanjang': {'durasi': str(gap_terpanjang),
                         'dari': str(dv.loc[ig-1,'waktu_pembacaan']), 'sampai': str(dv.loc[ig,'waktu_pembacaan'])},
    'dua_deret_utama': deret,
    'periode_awal': str(dv['waktu_pembacaan'].min()), 'periode_akhir': str(dv['waktu_pembacaan'].max()),
})
""")

    code("""
# -- QC NODE 3 ----------------------------------------------------
qc = {}
t = df_pasca6e['waktu_pembacaan']
d1 = t.diff() <= pd.Timedelta(minutes=2)        # hanya selisih antar-baris yang ~1 menit
for i, c in enumerate(kt_cols, 1):
    lonjakan_mentah = int(((df_pasca6e[c].diff().abs() > 20) & d1).sum())
    s = df_all['kelembaban_tanah_%d' % i]
    lonjakan_bersih = int(((s.diff().abs() > 20) & (df_all['waktu_pembacaan'].diff() <= pd.Timedelta(minutes=2))).sum())
    sub = df_all.loc[df_all['waktu_pembacaan'] >= '2026-09-30', 'kelembaban_tanah_%d' % i].dropna()
    qc[f'node{i}'] = {
        'jumlah_nol_mentah': int((df_raw[c] == 0).sum()),
        'lonjakan_abs_diff_gt20_mentah': lonjakan_mentah,
        'lonjakan_abs_diff_gt20_setelah_cleaning': lonjakan_bersih,
        'persen_waktu_ge_99_sejak_30sep': float((sub >= 99).mean()*100) if len(sub) else None,
        'n_sejak_30sep': int(len(sub)),
    }
sd3 = df_all.loc[df_all['waktu_pembacaan'] >= '2026-10-09', 'kelembaban_tanah_3'].dropna()
qc['node3_std_sejak_9okt'] = float(sd3.std()) if len(sd3) else None
qc['node3_mean_sejak_9okt'] = float(sd3.mean()) if len(sd3) else None
# Nilai tersering (modus) tiap node pada data mentah
qc['modus_mentah'] = {f'node{i}': float(df_raw[c].mode().iloc[0]) for i, c in enumerate(kt_cols, 1)}
print(json.dumps(qc, indent=2))
LAPORAN['qc_node3'] = qc
""")

    code("""
# -- CELL 8: Plot deret waktu -------------------------------------
gaps = dv['waktu_pembacaan'].diff()
gap_idx = gaps[gaps >= pd.Timedelta(hours=2)].index

def arsir(ax):
    for i in gap_idx:
        ax.axvspan(dv.loc[i-1,'waktu_pembacaan'], dv.loc[i,'waktu_pembacaan'], color='red', alpha=0.25, lw=0)

fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
for ax in axes: arsir(ax)
# garis diputus per segmen (tidak menyambung melintasi celah)
for _, g in dv.groupby('segmen'):
    axes[0].plot(g['waktu_pembacaan'], g['kelembaban_tanah_1'], lw=0.5, color='#1565C0')
    axes[0].plot(g['waktu_pembacaan'], g['kelembaban_tanah_2'], lw=0.5, color='#2E7D32')
    axes[1].plot(g['waktu_pembacaan'], g['kelembaban_tanah_avg'], lw=0.6, color='#E65100')
# Node 3: putus pada celah > GAP_MENIT di data node-3 yang valid
n3 = df_all[['waktu_pembacaan','kelembaban_tanah_3']].dropna()
for _, g in n3.groupby((n3['waktu_pembacaan'].diff() > pd.Timedelta(minutes=GAP_MENIT)).cumsum()):
    axes[2].plot(g['waktu_pembacaan'], g['kelembaban_tanah_3'], lw=0.5, color='gray')
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0],[0],color='#1565C0',label='Node 1'), Line2D([0],[0],color='#2E7D32',label='Node 2')], loc='lower left')
axes[0].set_title('Kelembaban tanah Node 1 dan Node 2 (valid)')
axes[1].set_title('Target: rata-rata Node 1-2')
axes[2].set_title('Node 3 (dikeluarkan dari pemodelan)')
for ax in axes: ax.set_ylim(0, 105); ax.set_ylabel('Kelembaban (%)')
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color='red', alpha=0.25, label='Tanpa data valid >= 2 jam')], loc='lower left')
plt.tight_layout()
plt.savefig(f'{OUT}/plot_timeseries_revisi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

    code("""
# -- CELL 9: Simpan -----------------------------------------------
final_cols = ['waktu_pembacaan','segmen'] + MODEL_COLS
dv[final_cols].to_csv(f'{OUT}/data_cleaned_revisi.csv', index=False)
LAPORAN['kolom_pemodelan'] = MODEL_COLS
with open(f'{OUT}/cleaning_report_revisi.json', 'w') as f:
    json.dump(LAPORAN, f, indent=2, default=str)
print(f'[OK] {OUT}/data_cleaned_revisi.csv  ({len(dv):,} baris, {dv["segmen"].nunique()} segmen)')
print(f'[OK] {OUT}/cleaning_report_revisi.json')
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '01r_preprocessing_sensor.ipynb')
print('ditulis, mode =', MODE)
