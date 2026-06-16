-- ============================================================================
-- 03-academic-profiling.sql
-- Tujuan: Memprofilkan data IPSIPK (performa akademik per semester).
-- ============================================================================

-- 3A. Jumlah record dan cakupan mahasiswa
SELECT COUNT(*) AS total, COUNT(DISTINCT Nim) AS unique_students
FROM dbo.IPSIPK;
-- Result: 6228 rows, 1605 unique students (16 mahasiswa tanpa data IPSIPK)

-- 3B. Cakupan untuk AP/IH saja (join dengan tblMHS)
SELECT COUNT(*) AS total, COUNT(DISTINCT i.Nim) AS students
FROM dbo.IPSIPK i
INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: 6228 rows, 1605 students

-- 3C. Sample record
SELECT TOP 10 Nim, Thn_Akademik, Periode, TSKS, ips, TTSKS, IPK, ThMasuk
FROM dbo.IPSIPK
ORDER BY Nim, Thn_Akademik, Periode;

-- 3D. Distribusi Periode (termasuk 'K' = semester pendek)
SELECT DISTINCT Periode FROM dbo.IPSIPK;
-- Result: '1' (Ganjil), '2' (Genap), 'K' (Pendek/Summer)

SELECT Periode, COUNT(*) AS jml FROM dbo.IPSIPK GROUP BY Periode;
-- Result: 1=~3000, 2=~3000, K=~200

-- 3E. Distribusi jumlah semester per mahasiswa
SELECT sem_count, COUNT(*) AS students
FROM (
    SELECT Nim, COUNT(*) AS sem_count
    FROM dbo.IPSIPK
    GROUP BY Nim
) t
GROUP BY sem_count
ORDER BY sem_count;
-- Result: 1=279, 2=163, 3=384, 4-12=779
-- → 458 mahasiswa punya <3 semester (angkatan baru, belum cukup data)

-- 3F. Mahasiswa dengan IPS NULL/0 di 4 semester pertama
WITH ranked AS (
    SELECT i.Nim, i.ips, i.Thn_Akademik, i.Periode,
           ROW_NUMBER() OVER (PARTITION BY i.Nim ORDER BY i.Thn_Akademik,
               CASE WHEN i.Periode = 'K' THEN '3' ELSE i.Periode END) AS rn
    FROM dbo.IPSIPK i
    INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
    WHERE m.Kode_jp IN ('AP','IH')
)
SELECT COUNT(*), SUM(CASE WHEN ips IS NULL OR ips = 0 THEN 1 ELSE 0 END)
FROM ranked
WHERE rn <= 4;
-- Result: 4873 records in first 4 sem, 1992 with ips NULL/0 (40.9%)

-- 3G. Profile angkatan tua (2014-2015) — banyak NULL
SELECT i.Nim, i.Thn_Akademik, i.Periode, i.TSKS, i.ips, i.TTSKS, i.IPK,
       m.ThMasuk, m.Status, m.Kode_jp
FROM dbo.IPSIPK i
INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
  AND (i.ips IS NULL)
ORDER BY i.Nim, i.Thn_Akademik, i.Periode;
-- Result: 2,489 records dengan ips NULL — mayoritas angkatan 2014-2015
-- Contoh: 2122085 (AP, 2014) punya 2 record IPSIPK, semua NULL.

-- 3H. Jumlah mahasiswa angkatan 2014-2015 dengan 0 IPS valid di 4 sem pertama
SELECT COUNT(DISTINCT m.Nim)
FROM dbo.tblMHS m
INNER JOIN (
    SELECT Nim, SUM(CASE WHEN ips IS NOT NULL THEN 1 ELSE 0 END) AS valid_ips
    FROM (
        SELECT i.Nim, i.ips, i.Thn_Akademik, i.Periode,
               ROW_NUMBER() OVER (PARTITION BY i.Nim ORDER BY i.Thn_Akademik, i.Periode) AS rn
        FROM dbo.IPSIPK i
        INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
        WHERE m.Kode_jp IN ('AP','IH')
    ) ranked
    WHERE rn <= 4
    GROUP BY Nim
    HAVING SUM(CASE WHEN ips IS NOT NULL THEN 1 ELSE 0 END) = 0
) null_ips ON m.Nim = null_ips.Nim;
-- Result: ~154 mahasiswa → excluded dari dataset
