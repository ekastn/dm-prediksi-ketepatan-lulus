# Table Mapping — Relasi dan Keputusan

Dokumentasi tabel mana yang digunakan dan mana yang ditolak, beserta alasannya.

## Tabel Digunakan (4 tabel)

```
┌──────────────┐
│   tblMHS     │  Data master mahasiswa
│   1621 rows  │  PK: Nim
└──────┬───────┘
       │ Nim
       │
       ├──────────< IPSIPK           (1:N)
       │              6228 rows, 1605 Nims
       │              IPS, IPK, SKS per semester
       │              → ips_sem1-4, ipk_sem4, sks_sem1-4, total_sks_lulus_sem4
       │
       ├──────────< Qnilai_mhs        (1:N)
       │              54587 rows, 1605 Nims
       │              Nilai MK + Kehadiran
       │              → failed_courses, repeated_courses, failed_in_sem1, avg_attendance
       │
       └──────────< Kul_Kehadiran     (1:N)
                     54587 rows, 1605 Nims
                     JmlHadir, maxpertemuan
                     → avg_attendance (supplement)
```

## Tabel Ditolak / Excluded

| Tabel | Rows | Reason |
|-------|------|--------|
| `HtblNilai` | 39,990 | **0% NIM overlap** dengan `tblMHS`. Format NIM berbeda (10-char legacy vs 9-digit). Data dari sistem informasi akademik versi lama. |
| `att_tblnilai` | 54,587 | Mirror dari `Qnilai_mhs` (data duplikat). Tidak memberikan nilai tambah. |
| `deltblNilai` | ? | Tabel deleted records. Data sudah tidak valid. |
| `feed_nilai` | 1,457 | Cakupan hanya 184 mahasiswa (11%). Kehadiran bagus (85% terisi) tapi sample terlalu kecil. |
| `feed_PerkuliahanMahasiswa` | 6,222 | Data duplikat dari `IPSIPK` (struktur mirip). Tidak menambah informasi baru. |
| `Perwalian` | 6,222 | `TSKSB` **semua NULL** (6222 rows, 0 non-null). Tidak bisa dipakai untuk SKS per semester. |
| `R_PWL` | 6,228 | Report perwalian. Ada data IPS/IPK untuk angkatan 2015, tapi hanya semester 2017-2019 (bukan semester awal). Coverage spotty. |
| `Luusan` | 0 | Tabel kosong. Data lulusan disimpan di `tblMHS.Status` bukan di tabel terpisah. |
| `tblCuti` | 0 | Tabel kosong. |
| `tblDet_DosenWali` | 0 | Tabel kosong. |
| `feed_mkprodi` | ? | Ada data SKS per MK, tapi Kode_MK tidak overlap dengan Qnilai_mhs untuk angkatan 2014-2015 → tidak bisa untuk komputasi IPS alternatif. |

## Format NIM antar Tabel

| Tabel | Format Contoh | Era |
|-------|--------------|-----|
| `tblMHS` | `207421001` | Post-2020 (9-digit) |
| `IPSIPK` | `207421001` | Post-2020 (9-digit) |
| `Qnilai_mhs` | `207421001` | Post-2020 (9-digit) |
| `Kul_Kehadiran` | `207421001` | Post-2020 (9-digit) |
| `feed_nilai` | `217421001` | Post-2021 (9-digit) |
| `HtblNilai` | `0301020026    ` | Pre-2020 (10-char padded) |
| `HtblNilai` (older) | `01.01.014.001` | Old format (dotted) |

## Keputusan Final

**Hanya 4 tabel yang digunakan** dalam pipeline ekstraksi:
1. `tblMHS` → identitas, demografi, status akhir
2. `IPSIPK` → performa per semester (IPS, IPK, SKS)
3. `Qnilai_mhs` → nilai MK + kehadiran
4. `Kul_Kehadiran` → kehadiran tambahan

Semua tabel lain ditolak karena: tidak ada data (tabel kosong), tidak ada overlap NIM (sistem berbeda), atau data duplikat (tidak menambah informasi).
