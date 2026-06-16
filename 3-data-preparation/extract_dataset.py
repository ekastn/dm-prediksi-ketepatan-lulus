"""
Dataset Extraction v2 — Prediksi Ketepatan Lulus Mahasiswa
===========================================================
Improvements over v1:
  - Lowered threshold: 3 semesters minimum (was 4)
  - Looser IPS filter: 1 valid IPS minimum (was 2)
  - New demographic features: jenis_kelamin, id_agama
  - New derived features: ips_std, ips_max, ips_min, failed_in_sem1
  - program (D3/S1) as explicit feature
  - Reclassify Aktif/Cuti beyond expected duration as target=0

Sumber: SQL Server (LITIGASI) — hasil migrasi vendor.
Usage: ../.venv/bin/python3 extract_dataset.py
"""
import pymssql
import csv
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "server": "localhost",
    "port": 1433,
    "user": "sa",
    "password": "MSSQL01#",
    "database": "LITIGASI",
}

PROGRAM_DURATION = {"IH": 8, "AP": 6}
EXPECTED_SKS_4_SEMESTERS = 80
MIN_SEMESTERS = 3  # lowered from 4
MIN_VALID_IPS = 1  # lowered from 2
OUTPUT_PATH = "/home/fzymorn/kuliah/semester-06/data-mining/prediksi-ketepatan-lulus/3-data-preparation/dataset.csv"


def parse_periode(p):
    """'1'=1, '2'=2, 'K'=3 (short/summer semester)."""
    if p is None:
        return 0
    p = p.strip()
    if p == "K":
        return 3
    try:
        return int(p)
    except ValueError:
        return 0


def connect():
    return pymssql.connect(**DB_CONFIG)


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_features(student, ipsipk_records, nilai_records, kehadiran_records):
    """
    Compute all features for one student.

    Returns dict of features + target, or target=None if outcome unknown.
    """
    f = {
        "student_id": student["nim"],
        "angkatan": student["th_masuk"],
        # --- New demographic features ---
        "program": student["kode_jp"],
        "jenis_kelamin": student["jenis_kel"],
        "id_agama": student["id_agama"],
    }

    # === Sort semesters chronologically ===
    ipsipk_sorted = sorted(ipsipk_records, key=lambda x: (x["thn_akademik"], x["periode"]))
    total_semesters = len(ipsipk_sorted)
    first_n = ipsipk_sorted[:4]  # up to 4 semesters for feature extraction

    # === IPS and SKS per semester (1-4), NULL if not available ===
    # Detect abnormal TSKS: some 2020+ angkatan have cumulative total instead of
    # per-semester values (database inconsistency, TSKS stored as program total ~131).
    # Check ALL first 4 semesters — some students have normal sem1 but bug sem2.
    abnormal_tsks = any(
        i < len(first_n)
        and first_n[i]["tsks"] is not None
        and first_n[i]["tsks"] > 30
        for i in range(min(len(first_n), 4))
    )

    for i in range(4):
        idx = i + 1
        # IPS: always from IPSIPK
        if i < len(first_n) and first_n[i]["ips"] is not None:
            f[f"ips_sem{idx}"] = first_n[i]["ips"]
        else:
            f[f"ips_sem{idx}"] = None

        # SKS: derive from Qnilai_mhs if TSKS is abnormal, else use IPSIPK
        if i < len(first_n):
            if abnormal_tsks:
                # Count distinct courses in this semester from Qnilai_mhs.
                # Note: Qnilai_mhs may also have bulk-loaded data (50+ MK/sem),
                # so cap at reasonable range (1-20); outliers → NULL → imputation.
                boundary = (first_n[i]["thn_akademik"], first_n[i]["periode"])
                prev_boundary = (
                    (first_n[i - 1]["thn_akademik"], first_n[i - 1]["periode"])
                    if i > 0 else ("", 0)
                )
                courses_in_sem = set(
                    n["kode_mk"] for n in nilai_records
                    if n["kode_mk"]
                    and (n["thn_akademik"], n["periode"]) <= boundary
                    and (
                        i == 0
                        or (n["thn_akademik"], n["periode"]) > prev_boundary
                    )
                )
                n_courses = len(courses_in_sem)
                # Cap at reasonable range: 1-20 courses per semester
                if 1 <= n_courses <= 20:
                    f[f"sks_sem{idx}"] = n_courses
                else:
                    f[f"sks_sem{idx}"] = None
            elif first_n[i]["tsks"] is not None:
                f[f"sks_sem{idx}"] = first_n[i]["tsks"]
            else:
                f[f"sks_sem{idx}"] = None
        else:
            f[f"sks_sem{idx}"] = None

    # === IPK and cumulative SKS at latest available semester (up to sem 4) ===
    last_idx = min(len(first_n), 4) - 1
    if last_idx >= 0:
        f["ipk_sem4"] = first_n[last_idx]["ipk"]
        f["total_sks_lulus_sem4"] = first_n[last_idx]["ttsks"]
    else:
        f["ipk_sem4"] = None
        f["total_sks_lulus_sem4"] = None

    # === Derived IPS features ===
    ips_vals = [r["ips"] for r in first_n if r["ips"] is not None]
    n_ips = len(ips_vals)

    if n_ips >= 2:
        f["avg_ips"] = round(statistics.mean(ips_vals), 4)
        f["ips_trend"] = round(ips_vals[-1] - ips_vals[0], 4)
        f["ips_std"] = round(statistics.stdev(ips_vals), 4) if n_ips >= 2 else None
        f["ips_max"] = round(max(ips_vals), 4)
        f["ips_min"] = round(min(ips_vals), 4)
    else:
        f["avg_ips"] = None
        f["ips_trend"] = None
        f["ips_std"] = None
        f["ips_max"] = None
        f["ips_min"] = None

    # === SKS completion ratio ===
    if last_idx >= 0 and first_n[last_idx]["ttsks"] is not None:
        f["sks_completion_ratio"] = round(first_n[last_idx]["ttsks"] / EXPECTED_SKS_4_SEMESTERS, 4)
    else:
        f["sks_completion_ratio"] = None

    # === Grade features (first 4 semesters of courses) ===
    sem_boundary = (
        first_n[last_idx]["thn_akademik"] if last_idx >= 0 else "",
        first_n[last_idx]["periode"] if last_idx >= 0 else 0,
    )

    first4_courses = [
        n for n in nilai_records
        if (n["thn_akademik"], n["periode"]) <= sem_boundary
    ]

    # Failed courses (E, D, T)
    f["failed_courses"] = sum(1 for c in first4_courses if c["nilai"] in ("E", "D", "T"))

    # Failed in semester 1 specifically (early warning signal)
    sem1_courses = [c for c in first4_courses if c["thn_akademik"] <= first_n[0]["thn_akademik"]
                    and c["periode"] <= first_n[0]["periode"]]
    f["failed_in_sem1"] = sum(1 for c in sem1_courses if c["nilai"] in ("E", "D", "T"))

    # Repeated courses: same Kode_MK appears >1 time in first 4 semesters
    course_count = defaultdict(int)
    for c in first4_courses:
        if c["kode_mk"]:
            course_count[c["kode_mk"]] += 1
    f["repeated_courses"] = sum(1 for v in course_count.values() if v > 1)

    # === Average attendance (Qnilai_mhs + Kul_Kehadiran) ===
    att_nilai = [c["kehadiran"] for c in first4_courses if c["kehadiran"] is not None]
    att_kuliah = [k["pct"] for k in kehadiran_records
                  if (k["thn_akademik"], k["periode"]) <= sem_boundary]
    all_att = att_nilai + att_kuliah
    f["avg_attendance"] = round(statistics.mean(all_att), 2) if all_att else None

    # === semester_count: how many semesters of data available ===
    f["semester_count"] = total_semesters

    # === Target ===
    expected = PROGRAM_DURATION.get(student["kode_jp"], 8)
    status = student["status"]

    if status in ("L", "Lulus"):
        # Graduated — check if within expected duration
        f["target"] = 1 if total_semesters <= expected else 0
    elif status == "Keluar":
        # Dropped out
        f["target"] = 0
    elif status in ("Aktif", "Cuti") and total_semesters > expected:
        # Still active but past expected duration → tidak tepat waktu
        f["target"] = 0
    else:
        # Aktif/Cuti within duration, A, D, -, NULL → unknown
        f["target"] = None

    return f


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    conn = connect()
    cur = conn.cursor()

    # === STEP 1: Fetch IPSIPK ===
    print("Fetching IPSIPK...")
    cur.execute("""
        SELECT i.Nim, i.Thn_Akademik, i.Periode, i.TSKS, i.ips, i.TTSKS, i.IPK
        FROM dbo.IPSIPK i
        INNER JOIN dbo.tblMHS m ON i.Nim = m.Nim
        WHERE m.Kode_jp IN ('AP','IH')
        ORDER BY i.Nim, i.Thn_Akademik, i.Periode
    """)
    ipsipk_by_nim = defaultdict(list)
    for r in cur.fetchall():
        ipsipk_by_nim[r[0].strip()].append({
            "thn_akademik": r[1].strip() if r[1] else "",
            "periode": parse_periode(r[2]),
            "tsks": int(r[3]) if r[3] is not None else None,
            "ips": float(r[4]) if r[4] is not None else None,
            "ttsks": int(r[5]) if r[5] is not None else None,
            "ipk": float(r[6]) if r[6] is not None else None,
        })
    print(f"  {len(ipsipk_by_nim)} students")

    # === STEP 2: Fetch Qnilai_mhs (grades + attendance) ===
    print("Fetching Qnilai_mhs...")
    cur.execute("""
        SELECT n.Nim, n.Kode_MK, n.Nilai, n.Angka, n.Kehadiran,
               n.Thn_Akademik, n.Periode
        FROM dbo.Qnilai_mhs n
        INNER JOIN dbo.tblMHS m ON n.Nim = m.Nim
        WHERE m.Kode_jp IN ('AP','IH')
        ORDER BY n.Nim, n.Thn_Akademik, n.Periode
    """)
    nilai_by_nim = defaultdict(list)
    for r in cur.fetchall():
        kehadiran = float(r[4]) if r[4] is not None and r[4] > 0 else None
        nilai_by_nim[r[0].strip()].append({
            "kode_mk": r[1].strip() if r[1] else "",
            "nilai": r[2].strip() if r[2] else "",
            "angka": float(r[3]) if r[3] is not None else None,
            "kehadiran": kehadiran,
            "thn_akademik": r[5].strip() if r[5] else "",
            "periode": parse_periode(r[6]),
        })
    print(f"  {len(nilai_by_nim)} students")

    # === STEP 3: Fetch Kul_Kehadiran (supplementary attendance) ===
    print("Fetching Kul_Kehadiran...")
    cur.execute("""
        SELECT k.Nim, k.Thn_Akademik, k.Periode, k.Kode_Mk,
               k.JmlHadir, k.maxpertemuan
        FROM dbo.Kul_Kehadiran k
        INNER JOIN dbo.tblMHS m ON k.Nim = m.Nim
        WHERE m.Kode_jp IN ('AP','IH')
          AND k.JmlHadir > 0 AND k.maxpertemuan > 0
        ORDER BY k.Nim, k.Thn_Akademik, k.Periode
    """)
    kehadiran_by_nim = defaultdict(list)
    for r in cur.fetchall():
        pct = round(float(r[4]) / float(r[5]) * 100, 2)
        kehadiran_by_nim[r[0].strip()].append({
            "thn_akademik": r[1].strip() if r[1] else "",
            "periode": parse_periode(r[2]),
            "kode_mk": r[3].strip() if r[3] else "",
            "pct": pct,
        })
    print(f"  {len(kehadiran_by_nim)} students with attendance")

    # === STEP 4: Fetch student master data ===
    print("Fetching students...")
    cur.execute("""
        SELECT Nim, Nama_Mhs, Kode_jp, ThMasuk, PerMasuk, Status, Status_Akd,
               Jenis_Kel, id_agama
        FROM dbo.tblMHS
        WHERE Kode_jp IN ('AP','IH')
    """)
    students = []
    for r in cur.fetchall():
        students.append({
            "nim": r[0].strip() if r[0] else "",
            "nama": r[1].strip() if r[1] else "",
            "kode_jp": r[2].strip() if r[2] else "",
            "th_masuk": r[3].strip() if r[3] else "",
            "per_masuk": r[4].strip() if r[4] else "",
            "status": r[5].strip() if r[5] else "",
            "status_akd": r[6].strip() if r[6] else "",
            "jenis_kel": r[7].strip() if r[7] else "",
            "id_agama": str(r[8]) if r[8] is not None else "",
        })
    conn.close()
    print(f"  {len(students)} students")

    # === STEP 5: Compute features per student ===
    rows = []
    skipped_short = 0
    skipped_null_ipsipk = 0
    skipped_unknown = 0

    for i, s in enumerate(students):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i + 1}/{len(students)}...")

        ipsipk = ipsipk_by_nim.get(s["nim"], [])

        # Filter 1: minimum semester threshold
        if len(ipsipk) < MIN_SEMESTERS:
            skipped_short += 1
            continue

        # Filter 2: minimum valid IPS records in first 4 semesters
        ipsipk_sorted = sorted(ipsipk, key=lambda x: (x["thn_akademik"], x["periode"]))
        valid_ips = sum(1 for r in ipsipk_sorted[:4] if r["ips"] is not None)
        if valid_ips < MIN_VALID_IPS:
            skipped_null_ipsipk += 1
            continue

        nilai = nilai_by_nim.get(s["nim"], [])
        kehadiran = kehadiran_by_nim.get(s["nim"], [])

        features = compute_features(s, ipsipk, nilai, kehadiran)
        if features["target"] is None:
            skipped_unknown += 1
            continue

        rows.append(features)

    # === STEP 6: Split into train/test by time ===
    # Train: angkatan <= SPLIT_YEAR (historical, outcomes matured)
    # Test:  angkatan >  SPLIT_YEAR (recent, simulates real prediction scenario)
    SPLIT_YEAR = 2021

    train_rows = [r for r in rows if int(r["angkatan"]) <= SPLIT_YEAR]
    test_rows = [r for r in rows if int(r["angkatan"]) > SPLIT_YEAR]

    # === STEP 7: Write CSV(s) ===
    fieldnames = [
        "student_id", "angkatan", "program",
        "jenis_kelamin", "id_agama",
        "ips_sem1", "ips_sem2", "ips_sem3", "ips_sem4",
        "ipk_sem4", "sks_sem1", "sks_sem2", "sks_sem3", "sks_sem4",
        "total_sks_lulus_sem4", "failed_courses", "failed_in_sem1",
        "repeated_courses", "avg_attendance",
        "ips_trend", "avg_ips", "ips_std", "ips_max", "ips_min",
        "sks_completion_ratio", "semester_count",
        "target",
    ]

    def write_csv(path, data):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k) for k in fieldnames})

    base = OUTPUT_PATH.replace(".csv", "")
    full_path = f"{base}.csv"
    train_path = f"{base}_train.csv"
    test_path = f"{base}_test.csv"

    write_csv(full_path, rows)
    write_csv(train_path, train_rows)
    write_csv(test_path, test_rows)

    # === STEP 8: Summary ===
    tepat = sum(1 for r in rows if r["target"] == 1)
    tidak = sum(1 for r in rows if r["target"] == 0)
    t1_tr = sum(1 for r in train_rows if r["target"] == 1)
    t0_tr = sum(1 for r in train_rows if r["target"] == 0)
    t1_ts = sum(1 for r in test_rows if r["target"] == 1)
    t0_ts = sum(1 for r in test_rows if r["target"] == 0)

    print(f"\n{'='*55}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*55}")
    print(f"Threshold: min {MIN_SEMESTERS} semesters, min {MIN_VALID_IPS} valid IPS")
    print(f"{'─'*55}")
    print(f"Total students in tblMHS (AP/IH):   {len(students)}")
    print(f"Skipped — < {MIN_SEMESTERS} semesters:              {skipped_short}")
    print(f"Skipped — too many NULL IPSIPK:       {skipped_null_ipsipk}")
    print(f"Skipped — unknown outcome:             {skipped_unknown}")
    print(f"{'─'*55}")
    print(f"FULL DATASET:                          {len(rows)} rows")
    print(f"  Tepat waktu (1):                     {tepat}")
    print(f"  Tidak tepat waktu (0):               {tidak}")
    print(f"  Imbalance:                           {tidak/(tepat+tidak)*100:.1f}% neg")
    print(f"{'─'*55}")
    print(f"TRAIN (angkatan <= {SPLIT_YEAR}):              {len(train_rows)} rows")
    print(f"  Tepat: {t1_tr}, Tidak: {t0_tr}  ({t0_tr/(t1_tr+t0_tr)*100:.1f}% neg)")
    print(f"TEST  (angkatan >  {SPLIT_YEAR}):              {len(test_rows)} rows")
    print(f"  Tepat: {t1_ts}, Tidak: {t0_ts}  ({t0_ts/(t1_ts+t0_ts)*100:.1f}% neg)")
    print(f"{'='*55}")
    print(f"\nOutputs:")
    print(f"  Full:  {full_path}")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")


if __name__ == "__main__":
    main()
