-- ============================================================================
-- 04-grade-profiling.sql
-- Tujuan: Memprofilkan tabel nilai (Qnilai_mhs vs HtblNilai vs feed_nilai).
-- ============================================================================

-- ============================================================
-- A. Qnilai_mhs (TABEL UTAMA untuk nilai + kehadiran)
-- ============================================================

-- 4A. Jumlah record dan cakupan
SELECT COUNT(*) AS total, COUNT(DISTINCT Nim) AS unique_students
FROM dbo.Qnilai_mhs;
-- Result: 54587 rows, 1605 unique

-- 4B. Cakupan untuk AP/IH
SELECT COUNT(*) AS total, COUNT(DISTINCT n.Nim) AS students
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: 54587 rows, 1605 students

-- 4C. Distribusi nilai huruf
SELECT Nilai, COUNT(*) AS jml
FROM dbo.Qnilai_mhs
GROUP BY Nilai
ORDER BY Nilai;
-- Result:
--   NULL=11, ' '=2, '-'=6575 (belum dinilai)
--   A=17787, A-=0? tidak terlihat, B=4707, B+=9012, B-=2259
--   C=613, C+=1063, C-=323
--   D=178, E=1130, T=188 (T = Tidak lulus)

-- 4D. Tabel nilai yang mungkin dari proposal ("tblNilai")
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME LIKE '%Nilai%' OR TABLE_NAME LIKE '%nilai%'
ORDER BY TABLE_NAME;
-- Result: att_tblnilai, deltblNilai, feed_nilai, FilterNilaiMK, HtblNilai, Qnilai_mhs, QNilaiAP

-- 4E. Analisis kehadiran di Qnilai_mhs
SELECT
    COUNT(*) AS total,
    COUNT(CASE WHEN Kehadiran IS NULL THEN 1 END) AS null_count,
    COUNT(CASE WHEN Kehadiran = 0 THEN 1 END) AS zero_count,
    COUNT(CASE WHEN Kehadiran > 0 THEN 1 END) AS positive_count
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: total=54587, null=13849, zero=30786, positive=9952
-- → Hanya 18.2% record punya kehadiran > 0

-- 4F. Sample nilai dengan kehadiran
SELECT TOP 10 n.Nim, n.Kode_MK, n.Nilai, n.Angka, n.Kehadiran,
       n.Thn_Akademik, n.Periode
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH') AND n.Kehadiran > 0
ORDER BY n.Nim, n.Thn_Akademik, n.Periode;

-- 4G. Distribusi Periode di Qnilai_mhs
SELECT DISTINCT Periode FROM dbo.Qnilai_mhs;
-- Result: '1', '2', 'K'

-- 4H. Qnilai_mhs untuk angkatan tua (2014-2015)
SELECT COUNT(DISTINCT n.Nim)
FROM dbo.Qnilai_mhs n
INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH') AND m.ThMasuk IN ('2014','2015');
-- Result: 313 students — data nilai tersedia untuk angkatan tua

-- ============================================================
-- B. HtblNilai (excluded: NIM format berbeda, 0% overlap)
-- ============================================================

-- 4I. Jumlah record HtblNilai
SELECT COUNT(*) AS total, COUNT(DISTINCT Nim) AS unique_students
FROM dbo.HtblNilai;
-- Result: 39990 rows, 1416 unique

-- 4J. Sample NIM dari HtblNilai
SELECT DISTINCT TOP 10 Nim FROM dbo.HtblNilai;
-- Result: '0301020026    ', '0301021001    ' — format berbeda (10-char, ada titik versi lama)

-- 4K. Overlap NIM antara HtblNilai dan tblMHS
SELECT COUNT(DISTINCT h.Nim)
FROM dbo.HtblNilai h
INNER JOIN dbo.tblMHS m ON h.Nim = m.Nim;
-- Result: 0 → TIDAK ADA OVERLAP. HtblNilai dari sistem berbeda (legacy).

-- 4L. Distribusi Kode_jp di HtblNilai (konfirmasi sistem berbeda)
SELECT Kode_Jp, COUNT(DISTINCT Nim) AS students
FROM dbo.HtblNilai
WHERE Kode_Jp IS NOT NULL
GROUP BY Kode_Jp
ORDER BY students DESC;
-- Result: T3=698, T1=277, P3=249, P1=167, H1-H10=15...
-- → Kode_jp format berbeda (T, P, H prefix), bukan AP/IH

-- ============================================================
-- C. feed_nilai (supplementary, coverage rendah)
-- ============================================================

-- 4M. Cakupan feed_nilai
SELECT COUNT(*) AS total, COUNT(DISTINCT Nim) AS unique_students
FROM dbo.feed_nilai;
-- Result: 1457 rows, 184 unique

-- 4N. Overlap dengan tblMHS
SELECT COUNT(DISTINCT f.Nim)
FROM dbo.feed_nilai f
INNER JOIN dbo.tblMHS m ON f.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: 184 students — semua match, tapi cuma 184 dari 1621

-- 4O. Kehadiran di feed_nilai
SELECT COUNT(*) AS total,
       COUNT(CASE WHEN Kehadiran IS NOT NULL AND Kehadiran > 0 THEN 1 END) AS with_hadir
FROM dbo.feed_nilai f
INNER JOIN dbo.tblMHS m ON f.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: total=1457, with_hadir=1233 → 84.6% punya kehadiran! Tapi cakupan mahasiswanya sedikit.

-- ============================================================
-- D. Kul_Kehadiran (supplementary attendance source)
-- ============================================================

-- 4P. Cakupan Kul_Kehadiran
SELECT COUNT(*) AS total,
       COUNT(CASE WHEN JmlHadir > 0 AND maxpertemuan > 0 THEN 1 END) AS valid_att,
       COUNT(DISTINCT CASE WHEN JmlHadir > 0 AND maxpertemuan > 0 THEN k.Nim END) AS students_with_att
FROM dbo.Kul_Kehadiran k
INNER JOIN dbo.tblMHS m ON k.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH');
-- Result: total=54587, valid_att=595, students=146
-- → Hanya 146 mahasiswa dengan data kehadiran valid

-- 4Q. Sample Kul_Kehadiran dengan data valid
SELECT TOP 5 k.Nim, k.Kode_Mk, k.Thn_Akademik, k.Periode,
       k.JmlHadir, k.maxpertemuan,
       CAST(k.JmlHadir AS FLOAT)/k.maxpertemuan * 100 AS pct
FROM dbo.Kul_Kehadiran k
INNER JOIN dbo.tblMHS m ON k.Nim = m.Nim
WHERE m.Kode_jp IN ('AP','IH')
  AND k.JmlHadir > 0 AND k.maxpertemuan > 0
ORDER BY k.Nim, k.Thn_Akademik, k.Periode;

-- ============================================================
-- E. Kesimpulan
-- ============================================================
-- Tabel nilai utama:  Qnilai_mhs (1605 students, 54587 records)
-- Tabel nilai excluded: HtblNilai (0% overlap NIM)
-- Supplement kehadiran: Kul_Kehadiran (146 students)
-- feed_nilai: 184 students, kehadiran bagus (85%) tapi cakupan kecil
