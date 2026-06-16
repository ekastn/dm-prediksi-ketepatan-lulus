# Data Understanding — Chronological Exploration Log

Log aktual eksplorasi database LITIGASI yang dilakukan secara iteratif.
Query dijalankan melalui dua medium: `docker exec sqlcmd` (CLI) dan Python `pymssql` (inline scripts).

---

## Round 1: "Apa isi database ini?"

### 1.1 Baca DDL
**Aksi:** Baca file `ddl.csv` — 2880 baris, mendokumentasikan semua kolom dari 405 tabel.

**Tools:** `grep` + `read` tool (bukan SQL)

**Temuan:** Ada tabel `tblMHS`, `IPSIPK`, `Perwalian`, `HtblNilai`, `Qnilai_mhs`, tapi tidak ada yang bernama persis `tblNilai`.

```bash
grep -i "tblMHS\|IPSIPK\|tblNilai\|Perwalian\|Luusan" ddl.csv
# tblMHS ditemukan di line 2781, IPSIPK di line 945, Perwalian di line 1450
# "tblNilai" TIDAK ADA — ini tabel imajiner dari proposal user
```

### 1.2 Coba konek ke SQL Server — GAGAL
**Aksi:** Query langsung ke container tanpa spesifikasi database.

```bash
docker exec sqlserver /opt/mssql-tools18/bin/sqlcmd -C -S localhost \
  -U sa -P 'MSSQL01#' \
  -Q "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN ('tblMHS','IPSIPK')"
# → 0 rows affected. LHO? Tabelnya ga ada?
```

**Debug:** Cek database apa aja yang ada.

```sql
SELECT name, database_id FROM sys.databases ORDER BY name;
-- LITIGASI (5), master, model, msdb, tempdb
```

**Fix:** Tambahin `-d LITIGASI`. Semua query setelah ini pakai flag ini.

```bash
docker exec sqlserver /opt/mssql-tools18/bin/sqlcmd -C -S localhost \
  -U sa -P 'MSSQL01#' -d LITIGASI \
  -Q "SELECT COUNT(*) FROM dbo.tblMHS"
# → 1621. Nah, baru bener.
```

---

## Round 2: Profiling tblMHS

### 2.1 Overview mahasiswa
**Aksi:** Python inline via pymssql.

```python
import pymssql
conn = pymssql.connect(server='localhost', port=1433, user='sa', password='MSSQL01#', database='LITIGASI')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM dbo.tblMHS')
# → 1621 mahasiswa
cur.execute('SELECT Kode_jp, COUNT(*) FROM dbo.tblMHS GROUP BY Kode_jp')
# AP = 386 (D3), IH = 1235 (S1)
cur.execute('SELECT ThMasuk, COUNT(*) FROM dbo.tblMHS GROUP BY ThMasuk ORDER BY ThMasuk')
# 2014=168, 2015=147, ... , 2025=258
cur.execute('SELECT Status, COUNT(*) FROM dbo.tblMHS GROUP BY Status')
# L=621, Lulus=177, Aktif=612, Keluar=155, Cuti=41, A/D/-/NULL=16
```

**Keputusan:** Ada dua nilai "lulus": `L` dan `Lulus`. `L` kayaknya dari sistem lama, `Lulus` dari sistem baru. Dua-duanya harus dianggap sebagai lulus.

### 2.2 Mapping Kode_jp ke Program
```python
cur.execute('SELECT Kode_Jp, Jurusan, Program FROM dbo.TblJurusan')
# AP → Administrasi Peradilan D3
# IH → Ilmu Hukum S1
# SKSTPL, ipkta, skskp semuanya NULL — sayang, ga bisa dipakai
```

---

## Round 3: Profiling IPSIPK

### 3.1 Struktur data
```python
cur.execute('SELECT TOP 5 * FROM dbo.IPSIPK ORDER BY Nim, Thn_Akademik, Periode')
# Nim, Thn_Akademik, Periode, TSKS, ips, TTSKS, IPK, Nama_Mhs, Kode_jp, Status_Akd, ThMasuk
```

### 3.2 ANOMALI: Periode 'K'
```python
cur.execute('SELECT DISTINCT Periode FROM dbo.IPSIPK')
# ['1', '2', 'K']
```
**Temuan:** `K` = semester pendek/summer. Bukan Ganjil (1) atau Genap (2). Harus disortir terpisah.

### 3.3 ANOMALI: Nilai IPS NULL masif
```python
cur.execute("""
SELECT COUNT(*), SUM(CASE WHEN ips IS NULL OR ips = 0 THEN 1 ELSE 0 END)
FROM dbo.IPSIPK WHERE ...
""")
# 6228 total, ~2489 NULL/0 — 40%!
```

**Investigate:** Siapa yang NULL?
```python
cur.execute("""
SELECT m.ThMasuk, COUNT(*), 
       SUM(CASE WHEN i.ips IS NULL THEN 1 ELSE 0 END) AS null_ips
FROM dbo.IPSIPK i JOIN dbo.tblMHS m ON i.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
GROUP BY m.ThMasuk ORDER BY m.ThMasuk
""")
# 2014: 100% NULL — sistem lama ga nyatet IPS per semester
# 2015: ~70% NULL
# 2016+: <5% NULL
```

**Kesimpulan:** Angkatan 2014-2015 data IPSIPK-nya cuma placeholder. Ini efek migrasi vendor.

---

## Round 4: Mencari "tblNilai"

### 4.1 Proposal bilang sumber "tblNilai" — tapi tabel ini ga ada
```python
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE '%Nilai%'")
# HtblNilai, deltblNilai, feed_nilai, att_tblnilai, Qnilai_mhs, FilterNilaiMK, QNilaiAP, QNilaiSumAP
```

### 4.2 Coba HtblNilai — KANDIDAT TERKUAT?
```python
cur.execute('SELECT COUNT(*) FROM dbo.HtblNilai')
# 39990 rows. BANYAK. Ini pasti tabel utama.
cur.execute('SELECT DISTINCT TOP 10 Nim FROM dbo.HtblNilai')
# '0301020026    ', '0301021001    ' — format 10-char dengan spasi
```

### 4.3 Cek overlap dengan tblMHS — SHOCK
```python
cur.execute("""
SELECT COUNT(DISTINCT h.Nim) FROM dbo.HtblNilai h
INNER JOIN dbo.tblMHS m ON h.Nim = m.Nim
""")
# 0. NOL. Tidak ada overlap sama sekali.
```

**Analisis:** `HtblNilai` berasal dari sistem informasi akademik versi lama. Format NIM-nya beda (10-char, terkadang pakai titik seperti `01.01.014.001`). Sedangkan `tblMHS` pakai format 9-digit (`207421001`) dari sistem baru. Dua sistem berbeda, NIM tidak bisa di-join.

### 4.4 Coba Qnilai_mhs — ALTERNATIF
```python
cur.execute('SELECT COUNT(*) FROM dbo.Qnilai_mhs')
# 54587 rows
cur.execute("""
SELECT COUNT(DISTINCT n.Nim) FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
""")
# 1605 students. YES! Ini yang kita pakai.
```

### 4.5 Profiling nilai di Qnilai_mhs
```python
cur.execute('SELECT Nilai, COUNT(*) FROM dbo.Qnilai_mhs GROUP BY Nilai ORDER BY Nilai')
# A=17787, B=4707, B+=9012, B-=2259, C=613, C+=1063, C-=323, D=178, E=1130
# T=188 (Tidak lulus)
# '- '=6575 (belum dinilai / KRS kosong)
```

### 4.6 Profiling kehadiran
```python
cur.execute("""
SELECT 
    COUNT(*) AS total,
    COUNT(CASE WHEN Kehadiran IS NULL THEN 1 END) AS null_count,
    COUNT(CASE WHEN Kehadiran = 0 THEN 1 END) AS zero_count,
    COUNT(CASE WHEN Kehadiran > 0 THEN 1 END) AS positive_count
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
""")
# total=54587, null=13849, zero=30786, positive=9952
# HANYA 18.2% record punya kehadiran > 0!
```

**Kesimpulan sementara:** `avg_attendance` bakal sparse banget. Tapi proposal udah nyebutin fitur ini, jadi tetep disertakan.

---

## Round 5: Mencari sumber kehadiran alternatif

### 5.1 Kul_Kehadiran
```python
cur.execute("""
SELECT COUNT(*), 
       COUNT(CASE WHEN JmlHadir > 0 AND maxpertemuan > 0 THEN 1 END)
FROM dbo.Kul_Kehadiran k
INNER JOIN dbo.tblMHS m ON k.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
""")
# total=54587, valid=595 → cuma 595 record dengan kehadiran valid
# unique students: 146
```

### 5.2 feed_nilai
```python
cur.execute("""
SELECT COUNT(DISTINCT f.Nim)
FROM dbo.feed_nilai f
INNER JOIN dbo.tblMHS m ON f.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
""")
# 184 students. Cakupan kecil.
```

**Keputusan:** Supplement `Qnilai_mhs.Kehadiran` dengan `Kul_Kehadiran`. `feed_nilai` cakupannya terlalu kecil (184 dari 1621).

---

## Round 6: Mencari SKS per semester

### 6.1 Perwalian — GAGAL
```python
cur.execute("""
SELECT COUNT(*), SUM(CASE WHEN TSKSB IS NOT NULL THEN 1 ELSE 0 END)
FROM dbo.Perwalian
""")
# 6222 rows, 0 non-null TSKSB. SEMUA NULL. Ga bisa dipakai.
```

### 6.2 IPSIPK.TSKS — BEKERJA
`TSKS` (Total SKS Semester) di `IPSIPK` terisi untuk angkatan 2016+.
Untuk angkatan 2014-2015, sama seperti `ips` — NULL.
Tapi setidaknya untuk mayoritas mahasiswa, data ini tersedia.

---

## Round 7: Mencari data tambahan untuk angkatan tua

### 7.1 R_PWL (Report Perwalian)
```python
cur.execute("""
SELECT TOP 5 * FROM dbo.R_PWL WHERE IPS IS NOT NULL
""")
# Ada data IPS/IPK tapi untuk tahun 2017-2019.
# Untuk angkatan 2015: 140 students punya R_PWL, tapi kebanyakan cuma 1-2 semester.
# Tidak cukup untuk menggantikan 4 semester yang dibutuhkan.
```

### 7.2 Qnilai_mhs untuk angkatan tua
```python
cur.execute("""
SELECT COUNT(DISTINCT n.Nim)
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH') AND m.ThMasuk IN ('2014','2015')
""")
# 313 students — data nilai TERSEDIA untuk angkatan tua
```

### 7.3 Bisa komputasi IPS dari Qnilai_mhs?
Butuh SKS per MK. Coba join dengan `feed_mkprodi`:
```python
cur.execute("""
SELECT COUNT(DISTINCT q.Kode_MK) AS q_mk,
       COUNT(DISTINCT CASE WHEN f.Kode_Mk IS NOT NULL THEN q.Kode_MK END) AS overlap
FROM dbo.Qnilai_mhs q
JOIN dbo.tblMHS m ON q.Nim = m.Nim
LEFT JOIN dbo.feed_mkprodi f ON q.Kode_MK = f.Kode_Mk AND m.Kode_jp = f.Kode_Jp
WHERE m.ThMasuk IN ('2014','2015')
""")
# overlap = 0. Kode MK tidak cocok antara Qnilai_mhs dan feed_mkprodi.
# Tidak bisa komputasi IPS alternatif.
```

**Kesimpulan final:** Angkatan 2014-2015 tidak bisa diselamatkan. Mereka akan di-exclude.

---

## Round 8: Deteksi fitur demografi yang usable

### 8.1 Cek fill rate
```python
cur.execute("""
SELECT 
    COUNT(*) AS total,
    COUNT(Jenis_Kel) AS jk,        -- 1621
    COUNT(JalurMasuk) AS jalur,     -- 0!
    COUNT(Gelombang) AS gel,        -- 0!
    COUNT(penerima_kps) AS kps,     -- 1621
    COUNT(id_agama) AS agama,       -- 1621
    COUNT(Status_Masuk) AS sm,      -- 1613
    COUNT(Konsentrasi) AS kons      -- 6!
FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
""")
```

### 8.2 Cek zero-variance
```python
for col in ['Jenis_Kel','JalurMasuk','Status_Masuk','penerima_kps',
             'id_jalur_masuk','id_jenis_daftar','id_agama']:
    cur.execute(f'SELECT DISTINCT {col} FROM dbo.tblMHS WHERE Kode_jp IN (\'AP\',\'IH\')')
    vals = [r[0] for r in cur.fetchall()]
    print(f'{col}: {vals}')
```

**Hasil:**
- `JalurMasuk`: [None] → DROP (all NULL)
- `Gelombang`: [None] → DROP (all NULL)
- `penerima_kps`: [0] → DROP (all 0)
- `id_jenis_daftar`: [1] → DROP (all 1)
- `id_jalur_masuk`: [12, 11, 3] → DROP (99.4% nilai 12)
- `Status_Masuk`: ['Baru', '1', None] → DROP (97% 'Baru')
- `Jenis_Kel`: ['L', 'P'] → KEEP
- `id_agama`: [1,2,3,4,99] → KEEP

---

## Round 9: First extraction attempt — GAGAL (timeout)

Script pertama query per-mahasiswa sequential. 1,400+ round trip ke database.
**Timeout di 120 detik.**

**Fix:** Rewrite dengan bulk query — ambil semua data ke memory Python, proses di local.

---

## Round 10: Extraction berhasil tapi data kurang

Versi pertama: 397 rows, min 4 semester, min 2 IPS valid.

User minta tambahan data → longgarkan threshold: min 3 semester, min 1 IPS valid.

Versi kedua: **608 rows**. Ditambah fitur baru: `program`, `jenis_kelamin`, `id_agama`, `ips_std`, `ips_max`, `ips_min`, `failed_in_sem1`, `semester_count`.

Fitur `status_masuk`, `penerima_kps`, `id_jenis_daftar`, `id_jalur_masuk` di-drop setelah verifikasi zero-variance di dataset final.

---

## Round 11: SKS Data Quality Bug (Post-Extraction Analysis)

### 11.1 Discovery
Dataset final menunjukkan `sks_sem1` max=133, `sks_sem2` max=99 — tidak mungkin
dalam 1 semester.

### 11.2 Investigation
Query ke IPSIPK:
- 207422012 (Angk 2020 IH): Sem#1 TSKS=131, TTSKS=131 → kumulatif, bukan per-semester
- 207422016 (Angk 2020 IH): Sem#1 TSKS=112, TTSKS=112 → kumulatif
- 2227421001 (Angk 2022 IH): Sem#1 TSKS=4, Sem#2 TSKS=97 → mix normal + kumulatif

310 baris di 4 semester pertama memiliki TSKS >30. Semua dari angkatan 2020+.

### 11.3 Attempted fix via Qnilai_mhs
Qnilai_mhs juga terkontaminasi: 207422012 punya 54 MK distinct di sem1.
Tidak bisa dipakai sebagai fallback langsung.

### 11.4 Solution
Deteksi: jika ada TSKS >30 di 4 semester pertama → flag abnormal.
Fallback: count distinct Kode_MK dari Qnilai_mhs, cap 1-20.
Di luar range → NULL → diimputasi (median per angkatan) di preprocessing.

---

## Timeline Summary

| Waktu (approx) | Aktivitas | Tool |
|----------------|-----------|------|
| 23:03 | Baca DDL, identifikasi kandidat tabel | `read`, `grep` |
| 23:04 | Koneksi ke SQL Server (gagal pertama, fix database) | `sqlcmd` |
| 23:06 | Profiling tblMHS: 1621 mhs, status, angkatan | `pymssql` inline |
| 23:08 | Profiling IPSIPK: struktur, Periode K, NULL rate | `pymssql` inline |
| 23:10 | Mencari "tblNilai": HtblNilai 0% overlap → Qnilai_mhs | `pymssql` inline |
| 23:12 | Profiling Qnilai_mhs: distribusi nilai, kehadiran sparse | `pymssql` inline |
| 23:14 | Cek Kul_Kehadiran, feed_nilai sebagai supplement | `pymssql` inline |
| 23:16 | Perwalian TSKSB NULL → pakai IPSIPK.TSKS | `pymssql` inline |
| 23:18 | Cek R_PWL, feed_mkprodi untuk selamatkan angkatan tua | `pymssql` inline |
| 23:19 | Deteksi zero-variance features | `pymssql` inline |
| 23:20 | Tulis extraction script v1 (sequential query) | `write` |
| 23:21 | TIMEOUT — rewrite dengan bulk query | `write` |
| 23:22 | Sukses: 397 rows | `bash` |
| 23:23 | Longgarkan threshold + tambah fitur → 608 rows | `edit` + `bash` |
| 23:24 | Final dataset + train/test split | `bash` |
| 23:39 | Dokumentasi ulang (folder ini) | `write` |
