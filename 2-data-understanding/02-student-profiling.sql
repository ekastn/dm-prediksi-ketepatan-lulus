-- ============================================================================
-- 02-student-profiling.sql
-- Tujuan: Memprofilkan data mahasiswa di tblMHS.
-- ============================================================================

-- 2A. Total mahasiswa
SELECT COUNT(*) AS total FROM dbo.tblMHS;
-- Result: 1621

-- 2B. Distribusi program studi (Kode_jp)
SELECT Kode_jp, COUNT(*) AS jml
FROM dbo.tblMHS
GROUP BY Kode_jp
ORDER BY Kode_jp;
-- Result: AP=386 (D3), IH=1235 (S1)

-- 2C. Program detail dari TblJurusan
SELECT Kode_Jp, Jurusan, Program, Alias, SKSTPL, ipkta, skskp
FROM dbo.TblJurusan;
-- Result: AP = Administrasi Peradilan D3, IH = Ilmu Hukum S1

-- 2D. Distribusi angkatan (ThMasuk)
SELECT ThMasuk AS angkatan, COUNT(*) AS jml
FROM dbo.tblMHS
GROUP BY ThMasuk
ORDER BY ThMasuk;
-- Result: 2014=168, 2015=147, ..., 2025=258

-- 2E. Distribusi status mahasiswa
SELECT Status, COUNT(*) AS jml
FROM dbo.tblMHS
GROUP BY Status
ORDER BY Status;
-- Result: L=621, Lulus=177, Aktif=612, Keluar=155, Cuti=41, A=11, D=1, -=1, NULL=2

-- 2F. Distribusi status per program
SELECT Status, Kode_jp, COUNT(*) AS jml
FROM dbo.tblMHS
GROUP BY Status, Kode_jp
ORDER BY Kode_jp, Status;
-- Result:
--   AP: L=302, Aktif=55, Keluar=25, Cuti=4
--   IH: L=319, Lulus=177, Aktif=557, Keluar=130, Cuti=37

-- 2G. Profil demografi (fill rate)
SELECT
    COUNT(*) AS total,
    COUNT(Jenis_Kel) AS jk_filled,
    COUNT(JalurMasuk) AS jalur_filled,      -- 0 → all NULL
    COUNT(Gelombang) AS gel_filled,           -- 0 → all NULL
    COUNT(Status_Masuk) AS sm_filled,
    COUNT(id_jalur_masuk) AS ijm_filled,
    COUNT(id_jenis_daftar) AS ijd_filled,
    COUNT(penerima_kps) AS kps_filled,
    COUNT(id_agama) AS agama_filled,
    COUNT(Konsentrasi) AS kons_filled         -- 6 → near-empty
FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH');

-- 2H. Distribusi nilai fitur demografi
SELECT Jenis_Kel, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY Jenis_Kel;
-- L=1079, P=542

SELECT id_agama, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY id_agama;
-- 1=1337, 2=255, 3=16, 4=12, 99=1

SELECT Status_Masuk, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY Status_Masuk;
-- Baru=1569, 1=44, NULL=8 → 97% sama (near zero-variance)

SELECT id_jalur_masuk, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY id_jalur_masuk;
-- 12=1611, 11=9, 3=1 → 99.4% nilai 12 (near zero-variance)

SELECT id_jenis_daftar, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY id_jenis_daftar;
-- 1=1621 → 100% sama (zero-variance)

SELECT penerima_kps, COUNT(*) FROM dbo.tblMHS
WHERE Kode_jp IN ('AP','IH') GROUP BY penerima_kps;
-- 0=1621 → 100% sama (zero-variance)

-- Keputusan: hanya Jenis_Kel dan id_agama yang dipakai.
-- Status_Masuk, id_jalur_masuk, id_jenis_daftar, penerima_kps, JalurMasuk, Gelombang dibuang.
