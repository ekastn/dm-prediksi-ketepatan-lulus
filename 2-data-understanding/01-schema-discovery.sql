-- ============================================================================
-- 01-schema-discovery.sql
-- Tujuan: Menemukan tabel yang relevan untuk ekstraksi fitur.
-- ============================================================================

-- 1A. Cari tabel kunci yang disebut di proposal (tblMHS, IPSIPK, Perwalian, tblNilai)
--     Note: tblNilai tidak ditemukan. Dicari alternatifnya.
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME IN ('tblMHS','IPSIPK','Perwalian','Luusan',
                      'HtblNilai','feed_nilai','Qnilai_mhs','Kul_Kehadiran')
ORDER BY TABLE_NAME;

-- 1B. Hitung total tabel di database
SELECT COUNT(*) AS table_count
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'dbo';
-- Result: 405

-- 1C. Verifikasi database aktif
SELECT name, database_id FROM sys.databases ORDER BY name;
-- Result: LITIGASI (id=5), master, model, msdb, tempdb

-- 1D. Cek struktur tblMHS (kolom kunci)
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'tblMHS' AND TABLE_SCHEMA = 'dbo'
ORDER BY ORDINAL_POSITION;

-- 1E. Cek apakah Perwalian punya data SKS (TSKSB)
SELECT COUNT(*) AS total_perwalian,
       SUM(CASE WHEN TSKSB IS NOT NULL THEN 1 ELSE 0 END) AS non_null_tsksb
FROM dbo.Perwalian;
-- Result: 6222 rows, 0 non-null TSKSB → tidak bisa dipakai untuk SKS

-- 1F. Cek R_PWL (perwalian report) sebagai alternatif IPS/SKS
SELECT TOP 5 Thn_Akademik, Periode, Nim, TSKS, IPS, IPK
FROM dbo.R_PWL
WHERE IPS IS NOT NULL
ORDER BY Nim, Thn_Akademik, Periode;

-- 1G. Cek apakah Luusan (tabel lulusan) punya data
SELECT COUNT(*) FROM dbo.Luusan;
-- Result: 0 → kosong. Data lulusan ada di tblMHS.Status

-- 1H. Cek feed_PerkuliahanMahasiswa (performa per semester)
SELECT TOP 5 Nim, Thn_Akademik, Periode, total_sks, ips, sks_semester, ipk
FROM dbo.feed_PerkuliahanMahasiswa;
SELECT COUNT(*) AS total, COUNT(DISTINCT Nim) AS unique_students
FROM dbo.feed_PerkuliahanMahasiswa;
-- Result: 6222 rows, 1606 unique students
