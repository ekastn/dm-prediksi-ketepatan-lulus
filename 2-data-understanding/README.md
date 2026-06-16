# Data Understanding Phase

Fase eksplorasi dan profiling database SQL Server `LITIGASI` untuk project prediksi ketepatan lulus mahasiswa.

## Catatan Kejujuran

File SQL di folder ini adalah **rekonstruksi** dari query yang dijalankan secara iteratif via Python `pymssql` (inline scripts), bukan SQL yang tersimpan rapi dari awal. Proses aslinya eksploratif — tiap query berikutnya ditentukan oleh hasil query sebelumnya. File SQL dibuat setelah eksplorasi selesai untuk keperluan dokumentasi dan reproduksibilitas.

Untuk kronologi asli beserta narasi dan keputusan yang diambil: [`exploration-log.md`](./exploration-log.md)

## Isi Folder

| File | Tipe | Isi |
|------|------|-----|
| `exploration-log.md` | Narasi kronologis | Log jujur langkah-demi-langkah eksplorasi, termasuk query yang gagal, iterasi, dan keputusan. |
| `01-data-profiling.ipynb` | Jupyter Notebook | Replay interaktif profiling data dengan output aktual. Bisa dijalankan ulang. |
| `01-schema-discovery.sql` | SQL | Query penemuan tabel kunci dari 405 tabel. |
| `02-student-profiling.sql` | SQL | Profiling `tblMHS`: status, angkatan, demografi. |
| `03-academic-profiling.sql` | SQL | Profiling `IPSIPK`: Periode K, NULL rate, distribusi semester. |
| `04-grade-profiling.sql` | SQL | Profiling `Qnilai_mhs` vs `HtblNilai` (excluded). |
| `05-data-quality.sql` | SQL | Deteksi zero-variance, NIM mismatch, anomali. |
| `06-table-mapping.md` | Dokumentasi | Mapping relasi tabel + tabel yang ditolak beserta alasan. |

## Database Overview

| Properti | Nilai |
|----------|-------|
| **Engine** | SQL Server 2022 (Docker: `mcr.microsoft.com/mssql/server:2022-latest`) |
| **Database** | `LITIGASI` |
| **Total tabel** | 405 (schema `dbo`) |
| **Mahasiswa di `tblMHS`** | 1,621 |

## Timeline Eksplorasi

| Step | Query File | Temuan Utama |
|------|-----------|-------------|
| 1 | [`01-schema-discovery.sql`](./01-schema-discovery.sql) | Ditemukan 4 tabel kunci: `tblMHS`, `IPSIPK`, `Qnilai_mhs`, `Kul_Kehadiran`. `HtblNilai` tidak kompatibel (format NIM berbeda). `Perwalian` TSKSB semua NULL. |
| 2 | [`02-student-profiling.sql`](./02-student-profiling.sql) | 1,621 mahasiswa. 2 program: AP (D3, 386) + IH (S1, 1,235). Status: L=302, Lulus=177, Aktif=612, Keluar=155, Cuti=41. |
| 3 | [`03-academic-profiling.sql`](./03-academic-profiling.sql) | IPSIPK: 6,228 record, 1,605 mahasiswa. Periode `K` ditemukan (semester pendek). Angkatan 2014-2015 mayoritas NULL (data tidak tercatat di sistem lama). |
| 4 | [`04-grade-profiling.sql`](./04-grade-profiling.sql) | Qnilai_mhs: 54,587 record, 1,605 mahasiswa. Nilai: A=17,787, B=28,010, E=1,130, D=49. Kehadiran: hanya 18% record yang punya nilai >0. |
| 5 | [`05-data-quality.sql`](./05-data-quality.sql) | Data quality issues teridentifikasi: IPSIPK NULL untuk angkatan tua, kehadiran sparse, format NIM tidak seragam antar tabel, fitur zero-variance terdeteksi. |
| 6 | [`06-table-mapping.md`](./06-table-mapping.md) | Mapping relasi antar tabel. Kesimpulan: 4 tabel digunakan, `HtblNilai` dan `tblCuti` excluded. |

## Keputusan Kunci dari Data Understanding

| Keputusan | Alasan |
|-----------|--------|
| **Gunakan `Qnilai_mhs` sebagai tabel nilai**, bukan `HtblNilai` | `HtblNilai` 0% overlap NIM dengan `tblMHS` (format berbeda, sistem legacy) |
| **Gunakan `IPSIPK` sebagai sumber IPS/SKS** | Data per semester paling lengkap |
| **Supplement `Kul_Kehadiran` untuk kehadiran** | `Qnilai_mhs.Kehadiran` hanya terisi 18% |
| **Exclude angkatan 2014 dari dataset** | Data IPSIPK seluruhnya NULL untuk angkatan ini |
| **Periode `K` di-map ke nilai 3** | Short/summer semester, disortir setelah semester genap |
| **Threshold final: min 3 semester, min 1 IPS valid** | Kompromi antara cakupan data vs kualitas fitur |
| **Split train/test berdasarkan waktu** | Angkatan ≤2021 train, >2021 test; menghindari data leakage |

## Data Quality Issues Teridentifikasi

| Issue | Dampak | Penanganan |
|-------|--------|------------|
| IPSIPK NULL untuk angkatan 2014-2015 | 154 mahasiswa dikeluarkan | Filter min 1 valid IPS |
| Kehadiran 53% missing | Fitur `avg_attendance` sparse | Dibiarkan NULL, model handle missing values |
| Format NIM tidak seragam | Join antar tabel gagal | Hanya gunakan tabel dengan NIM format sama |
| Fitur zero-variance (`status_masuk`, `penerima_kps`, dll) | Tidak informatif untuk model | Dibuang dari dataset |
| `Perwalian.TSKSB` seluruhnya NULL | Tidak bisa dipakai untuk SKS per semester | Gunakan `IPSIPK.TSKS` sebagai gantinya |
| 401 mahasiswa Aktif/Cuti | Outcome belum diketahui | Dikeluarkan dari dataset |

## Tools yang Digunakan

- **Koneksi DB**: `docker exec sqlserver /opt/mssql-tools18/bin/sqlcmd` + Python `pymssql`
- **IDE Query**: DBeaver (terhubung ke localhost:1433)
- **Profiling**: Python (`pymssql` + `csv` + `statistics`)
