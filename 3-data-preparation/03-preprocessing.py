import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
df = pd.read_csv(BASE / 'dataset.csv')
print(f"Loaded: {df.shape}")

# ─────────────────────────────────────────────
# Step 2: Replace system zeros in IPS/SKS
# ─────────────────────────────────────────────
ips_cols = ['ips_sem1', 'ips_sem2', 'ips_sem3', 'ips_sem4']
sks_cols = ['sks_sem1', 'sks_sem2', 'sks_sem3', 'sks_sem4']

print("\n=== Step 2: Replace system zeros (0.0 -> NaN) ===")
for col in ips_cols + sks_cols:
    zeros = (df[col] == 0.0).sum()
    df.loc[df[col] == 0.0, col] = np.nan
    if zeros > 0:
        print(f"  {col}: {zeros} zeros -> NaN")

# ─────────────────────────────────────────────
# Step 3: Create has_attendance + drop avg_attendance
# ─────────────────────────────────────────────
print("\n=== Step 3: Create has_attendance flag ===")
df['has_attendance'] = df['avg_attendance'].notna().astype(int)
att_present = df['has_attendance'].sum()
att_missing = len(df) - att_present
print(f"  has_attendance: {att_present} present, {att_missing} missing -> flagged")

# ─────────────────────────────────────────────
# Step 4: Drop redundant/leakage features
# ─────────────────────────────────────────────
print("\n=== Step 4: Drop redundant/leakage features ===")
drop_cols = ['student_id', 'ips_sem4', 'sks_sem4', 'ipk_sem4',
             'semester_count', 'ips_max', 'total_sks_lulus_sem4', 'avg_attendance']
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped {len(drop_cols)} columns. Shape: {df.shape}")

# ─────────────────────────────────────────────
# Step 5: Median imputation per angkatan
# ─────────────────────────────────────────────
print("\n=== Step 5: Median imputation per angkatan ===")
impute_cols = ['ips_sem1', 'ips_sem2', 'ips_sem3', 'sks_sem1', 'sks_sem2', 'sks_sem3']
for col in impute_cols:
    n_before = df[col].isnull().sum()
    df[col] = df.groupby('angkatan')[col].transform(lambda x: x.fillna(x.median()))
    n_after = df[col].isnull().sum()
    print(f"  {col}: {n_before} imputed, {n_after} remaining")

# Verify no remaining NULLs
remaining_nulls = df[impute_cols].isnull().sum().sum()
print(f"\n  Remaining NULLs in imputed cols: {remaining_nulls}")
if remaining_nulls > 0:
    print("  WARNING: Still have NULLs after imputation! Checking per column:")
    print(df[impute_cols].isnull().sum())

# ─────────────────────────────────────────────
# Step 6: Recompute derived features
# ─────────────────────────────────────────────
print("\n=== Step 6: Recompute derived features ===")

# ips_trend: last valid minus first valid
ips_have = df[['ips_sem1', 'ips_sem2', 'ips_sem3']]
df['ips_trend'] = ips_have.bfill(axis=1).iloc[:, -1] - ips_have.ffill(axis=1).iloc[:, 0]

# avg_ips
df['avg_ips'] = df[['ips_sem1', 'ips_sem2', 'ips_sem3']].mean(axis=1)

# ips_std
df['ips_std'] = df[['ips_sem1', 'ips_sem2', 'ips_sem3']].std(axis=1)

# ips_min
df['ips_min'] = df[['ips_sem1', 'ips_sem2', 'ips_sem3']].min(axis=1)

# sks_completion_ratio
df['sks_completion_ratio'] = (df['sks_sem1'] + df['sks_sem2'] + df['sks_sem3']) / 60

print(f"  ips_trend     : min={df['ips_trend'].min():.4f}, max={df['ips_trend'].max():.4f}")
print(f"  avg_ips       : min={df['avg_ips'].min():.4f}, max={df['avg_ips'].max():.4f}")
print(f"  ips_std       : min={df['ips_std'].min():.4f}, max={df['ips_std'].max():.4f}")
print(f"  ips_min       : min={df['ips_min'].min():.4f}, max={df['ips_min'].max():.4f}")
print(f"  sks_completion_ratio: min={df['sks_completion_ratio'].min():.4f}, max={df['sks_completion_ratio'].max():.4f}")

# ─────────────────────────────────────────────
# Step 7: Encode categorical features
# ─────────────────────────────────────────────
print("\n=== Step 7: Encode categorical features ===")
df['program'] = df['program'].map({'AP': 0, 'IH': 1})
df['jenis_kelamin'] = df['jenis_kelamin'].map({'L': 0, 'P': 1})
# id_agama stays as-is (1/2/4)
print(f"  program       : {df['program'].unique()}")
print(f"  jenis_kelamin : {df['jenis_kelamin'].unique()}")
print(f"  id_agama      : {sorted(df['id_agama'].unique())}")

# ─────────────────────────────────────────────
# Step 8: Final verification
# ─────────────────────────────────────────────
print("\n=== Step 8: Final verification ===")
print(f"  Shape        : {df.shape}")
print(f"  Columns      : {list(df.columns)}")
print(f"  Total NULLs  : {df.isnull().sum().sum()}")
if df.isnull().sum().sum() > 0:
    print("\n  NULLs per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\n  Target distribution:")
print(f"    {df['target'].value_counts().to_dict()}")
print(f"  Target % : {df['target'].mean()*100:.2f}% on-time")
print(f"\n  Dtypes:")
print(df.dtypes.to_string())
print(f"\n  Sample rows:")
print(df.sample(3, random_state=42).to_string(index=False))

# ─────────────────────────────────────────────
# Step 9: Reorder columns to match expected output
# ─────────────────────────────────────────────
print("\n=== Step 9: Reorder columns ===")
expected_order = [
    'angkatan', 'program', 'jenis_kelamin', 'id_agama',
    'ips_sem1', 'ips_sem2', 'ips_sem3',
    'sks_sem1', 'sks_sem2', 'sks_sem3',
    'failed_courses', 'failed_in_sem1', 'repeated_courses',
    'has_attendance',
    'ips_trend', 'avg_ips', 'ips_std', 'ips_min', 'sks_completion_ratio',
    'target'
]
# Only keep columns that exist in the dataframe
actual_order = [c for c in expected_order if c in df.columns]
df = df[actual_order]
print(f"  Columns reordered to: {list(df.columns)}")

# ─────────────────────────────────────────────
# Step 10: Save output
# ─────────────────────────────────────────────
output_path = BASE / 'dataset_clean.csv'
df.to_csv(output_path, index=False)
print(f"\n=== Step 10: Saved ===")
print(f"  Output: {output_path} ({df.shape[0]} rows x {df.shape[1]} cols)")
print("  Done.")
