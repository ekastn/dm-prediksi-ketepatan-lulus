-- ============================================================================
-- 05-data-quality.sql
-- Tujuan: Identifikasi data quality issues & anomali.
-- ============================================================================

-- ============================================================
-- A. NIM Format Mismatch antar tabel
-- ============================================================

-- 5A. Bandingkan format NIM antar tabel
SELECT TOP 5 Nim FROM dbo.tblMHS;
-- Format: '207421001' (9-digit, post-2020)

SELECT TOP 5 Nim FROM dbo.HtblNilai;
-- Format: '0301020026    ' (10-char dengan padding, pre-2020)
--          '01.01.014.001' (format titik, era lama)

-- 5B. Konfirmasi tidak ada overlap
SELECT COUNT(DISTINCT h.Nim)
FROM dbo.HtblNilai h
INNER JOIN dbo.tblMHS m ON h.Nim = m.Nim;
-- Result: 0 — confirmed, different student populations

-- ============================================================
-- B. IPSIPK Data Completeness
-- ============================================================

-- 5C. NULL analysis per kolom IPSIPK
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN ips IS NULL THEN 1 ELSE 0 END) AS null_ips,
    SUM(CASE WHEN TSKS IS NULL THEN 1 ELSE 0 END) AS null_tsks,
    SUM(CASE WHEN IPK IS NULL THEN 1 ELSE 0 END) AS null_ipk,
    SUM(CASE WHEN TTSKS IS NULL THEN 1 ELSE 0 END) AS null_ttsks
FROM dbo.IPSIPK i
INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: 6228 total, ~2489 null_ips (40%)

-- 5D. Angkatan mana yang paling banyak NULL
SELECT m.ThMasuk,
       COUNT(*) AS total_records,
       SUM(CASE WHEN i.ips IS NULL THEN 1 ELSE 0 END) AS null_ips,
       CAST(SUM(CASE WHEN i.ips IS NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 AS null_pct
FROM dbo.IPSIPK i
INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
GROUP BY m.ThMasuk
ORDER BY m.ThMasuk;
-- Result: 2014=100%, 2015=~70%, 2016=~20%, 2017+=<5%

-- ============================================================
-- C. Zero-Variance Feature Detection
-- ============================================================

-- 5E. Deteksi fitur dengan near-zero variance di tblMHS
SELECT 'JalurMasuk' AS col, COUNT(DISTinct JalurMasuk) AS unique_vals, COUNT(*) AS total
FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'Gelombang', COUNT(DISTinct Gelombang), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'penerima_kps', COUNT(DISTinct penerima_kps), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'id_jenis_daftar', COUNT(DISTinct id_jenis_daftar), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'id_jalur_masuk', COUNT(DISTinct id_jalur_masuk), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'Status_Masuk', COUNT(DISTinct Status_Masuk), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH')
UNION ALL
SELECT 'Konsentrasi', COUNT(DISTinct Konsentrasi), COUNT(*) FROM dbo.tblMHS WHERE Kode_jp IN ('AP','IH');
-- Results:
--   JalurMasuk:     0 unique / 1621 → drop (all NULL)
--   Gelombang:      0 unique / 1621 → drop (all NULL)
--   penerima_kps:   1 unique / 1621 → drop (all 0)
--   id_jenis_daftar:1 unique / 1621 → drop (all 1)
--   id_jalur_masuk: 3 unique / 1621 → drop (99.4% nilai 12)
--   Status_Masuk:   3 unique / 1621 → drop (97% "Baru")
--   Konsentrasi:    3 unique / 1621 → drop (hanya 6 non-NULL)

-- ============================================================
-- D. Target Class Distribution Check
-- ============================================================

-- 5F. Distribusi target potensial (status L/Lulus/Keluar dengan 3+ semester)
SELECT m.Status, m.Kode_jp, COUNT(DISTINCT m.Nim) AS jml
FROM dbo.tblMHS m
INNER JOIN dbo.IPSIPK i ON m.Nim = i.Nim
WHERE m.Kode_jp IN ('AP','IH')
  AND m.Status IN ('L','Lulus','Keluar')
  AND i.ips IS NOT NULL
  AND m.Nim IN (
    SELECT Nim FROM dbo.IPSIPK GROUP BY Nim HAVING COUNT(*) >= 3
  )
GROUP BY m.Status, m.Kode_jp
ORDER BY m.Kode_jp, m.Status;
-- Result: ~600 students with known outcome + sufficient data

-- ============================================================
-- E. Anomali: Periode 'K' (Short/Summer Semester)
-- ============================================================

-- 5G. Berapa banyak record IPSIPK dengan Periode 'K'
SELECT COUNT(*) FROM dbo.IPSIPK WHERE Periode = 'K';
SELECT COUNT(*) FROM dbo.Qnilai_mhs WHERE Periode = 'K';

-- 5H. Sample IPSIPK dengan Periode 'K'
SELECT TOP 5 * FROM dbo.IPSIPK WHERE Periode = 'K';

-- ============================================================
-- F. Aktif/Cuti Students Beyond Expected Duration
-- ============================================================

-- 5I. Mahasiswa Aktif/Cuti yang sudah melebihi durasi normal
SELECT m.Status, m.Kode_jp, COUNT(DISTinct m.Nim) AS jml
FROM dbo.tblMHS m
INNER JOIN (
    SELECT Nim, COUNT(*) AS total_sem
    FROM dbo.IPSIPK GROUP BY Nim
) i ON m.Nim = i.Nim
WHERE m.Kode_jp IN ('AP','IH')
  AND m.Status IN ('Aktif','Cuti')
  AND (
    (m.Kode_jp = 'IH' AND i.total_sem > 8) OR
    (m.Kode_jp = 'AP' AND i.total_sem > 6)
  )
GROUP BY m.Status, m.Kode_jp;
-- Result: Hanya 2 mahasiswa AP yang clearly beyond duration
