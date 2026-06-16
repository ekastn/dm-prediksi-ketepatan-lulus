#!/usr/bin/env python3
"""
EDA: Prediksi Ketepatan Lulus Mahasiswa

Replicates the analysis from 02-eda.ipynb but saves all charts as PNG files
instead of inline display.

Target: 1 = Tepat Waktu, 0 = Tidak Tepat
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Output directory for charts
chart_dir = os.path.join(os.path.dirname(__file__), 'eda-charts')
os.makedirs(chart_dir, exist_ok=True)

def save_chart(fig, name):
    """Save a figure to the charts directory."""
    path = os.path.join(chart_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {name}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset.csv'))
print(f'Shape: {df.shape}')
print()

# Drop student_id (not a feature)
df_eda = df.drop(columns=['student_id'])

# Separate feature groups
cat_cols = ['program', 'jenis_kelamin', 'id_agama']
ips_cols = ['ips_sem1', 'ips_sem2', 'ips_sem3', 'ips_sem4']
sks_cols = ['sks_sem1', 'sks_sem2', 'sks_sem3', 'sks_sem4']
grade_cols = ['failed_courses', 'failed_in_sem1', 'repeated_courses']
derived_cols = [
    'ips_trend', 'avg_ips', 'ips_std', 'ips_max', 'ips_min',
    'sks_completion_ratio', 'semester_count', 'avg_attendance'
]
meta_cols = ['angkatan', 'ipk_sem4', 'total_sks_lulus_sem4']
target_col = 'target'

# ===========================================================================
# Section 1: Structure & Types
# ===========================================================================
print("=" * 60)
print("SECTION 1: Structure & Types")
print("=" * 60)
df.info()
print()
print(df.describe(include='all').T)
print()

# ===========================================================================
# Section 2: Missing Values
# ===========================================================================
print("=" * 60)
print("SECTION 2: Missing Values")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'Count': missing, '%': missing_pct})
print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False))
print()

# Chart 01: missingno matrix + bar
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
msno.matrix(df_eda, ax=axes[0], sparkline=False, fontsize=10)
axes[0].set_title('Missing Value Matrix', fontsize=13, pad=15)

msno.bar(df_eda, ax=axes[1], fontsize=10)
axes[1].set_title('Missing Value Count per Column', fontsize=13, pad=15)
plt.tight_layout()
save_chart(fig, '01_missing_matrix.png')

# Missing by angkatan
missing_by_angkatan = df_eda.groupby('angkatan', observed=False).apply(
    lambda g: g[ips_cols].isnull().sum(axis=1).mean()
)
print('Rata-rata missing IPS per mahasiswa, by angkatan:')
print(missing_by_angkatan)
print()

# Missing by semester_count
missing_by_sem = df_eda.groupby('semester_count', observed=False).apply(
    lambda g: g[ips_cols].isnull().sum(axis=1).mean()
)
print('Rata-rata missing IPS per mahasiswa, by semester_count:')
print(missing_by_sem)
print()

# ===========================================================================
# Section 3: Target Variable
# ===========================================================================
print("=" * 60)
print("SECTION 3: Target Variable")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

target_counts = df['target'].value_counts()
axes[0].bar(
    ['Tidak Tepat (0)', 'Tepat Waktu (1)'],
    target_counts.values,
    color=['#E74C3C', '#2ECC71']
)
axes[0].set_title(f'Target Distribution (n={len(df)})', fontsize=13)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11)

ct = pd.crosstab(df['program'], df['target'], margins=True)
ct_pct = pd.crosstab(df['program'], df['target'], normalize='index') * 100
ct_pct.plot(kind='bar', ax=axes[1], color=['#E74C3C', '#2ECC71'])
axes[1].set_title('Target Distribution by Program', fontsize=13)
axes[1].set_ylabel('%')
axes[1].legend(['Tidak Tepat', 'Tepat Waktu'])

ct_angkatan = pd.crosstab(df['angkatan'], df['target'])
ct_angkatan.plot(kind='bar', stacked=True, ax=axes[2], color=['#E74C3C', '#2ECC71'])
axes[2].set_title('Target Distribution by Angkatan', fontsize=13)
axes[2].legend(['Tidak Tepat', 'Tepat Waktu'])

plt.tight_layout()
save_chart(fig, '02_target_distribution.png')

print('Crosstab: program x target')
print(ct)
print()
print('Percentages by program:')
print(ct_pct.round(1))
print()

# ===========================================================================
# Section 4: Categorical Features vs Target
# ===========================================================================
print("=" * 60)
print("SECTION 4: Categorical Features vs Target")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, cat_cols):
    ct_cat = pd.crosstab(df[col], df['target'], normalize='index') * 100
    ct_cat.plot(kind='bar', ax=ax, color=['#E74C3C', '#2ECC71'])
    ax.set_title(f'{col} vs Target', fontsize=12)
    ax.set_ylabel('%')
    ax.set_xlabel(col)
    ax.legend(['Tidak Tepat', 'Tepat Waktu'])

plt.tight_layout()
save_chart(fig, '03_categorical_vs_target.png')

# ===========================================================================
# Section 5: IPS Features Distribution
# ===========================================================================
print("=" * 60)
print("SECTION 5: IPS Features Distribution")
print("=" * 60)

# Chart 04: 2x2 grid histogram IPS sem1-4 with mean/median lines
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(ips_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=30, alpha=0.7, color='#3498DB', edgecolor='white')
    axes[i].axvline(
        data.median(), color='#E74C3C', linestyle='--',
        label=f'Median={data.median():.2f}'
    )
    axes[i].axvline(
        data.mean(), color='#F39C12', linestyle='--',
        label=f'Mean={data.mean():.2f}'
    )
    axes[i].set_title(
        f'{col} (n={len(data)}, missing={df[col].isnull().sum()})',
        fontsize=12
    )
    axes[i].legend(fontsize=9)

plt.suptitle('IPS Distribution per Semester', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '04_ips_distribution.png')

# Chart 05: 2x2 grid boxplot IPS by program
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(ips_cols):
    df.boxplot(column=col, by='program', ax=axes[i])
    axes[i].set_title(f'{col} by Program', fontsize=12)
    axes[i].set_xlabel('Program')

plt.suptitle('IPS Boxplot by Program', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '05_ips_by_program.png')

# Chart 06: 2x2 grid histogram IPS by target
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(ips_cols):
    for target_val, color, label in [
        (0, '#E74C3C', 'Tidak Tepat'), (1, '#2ECC71', 'Tepat Waktu')
    ]:
        data = df[df['target'] == target_val][col].dropna()
        axes[i].hist(data, bins=25, alpha=0.5, color=color, label=label)
    axes[i].set_title(f'{col} by Target', fontsize=12)
    axes[i].legend()

plt.suptitle('IPS Distribution vs Target', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '06_ips_by_target.png')

# ===========================================================================
# Section 6: SKS Features
# ===========================================================================
print("=" * 60)
print("SECTION 6: SKS Features")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(sks_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=25, alpha=0.7, color='#9B59B6', edgecolor='white')
    axes[i].set_title(
        f'{col} (n={len(data)}, missing={df[col].isnull().sum()})',
        fontsize=12
    )

plt.suptitle('SKS Distribution per Semester', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '07_sks_distribution.png')

# ===========================================================================
# Section 7: Grade Features
# ===========================================================================
print("=" * 60)
print("SECTION 7: Grade Features")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, grade_cols):
    for target_val, color, label in [
        (0, '#E74C3C', 'Tidak Tepat'), (1, '#2ECC71', 'Tepat Waktu')
    ]:
        data = df[df['target'] == target_val][col]
        ax.hist(data, bins=15, alpha=0.5, color=color, label=label)
    ax.set_title(f'{col} by Target', fontsize=12)
    ax.set_xlabel(col)
    ax.legend()

plt.suptitle('Grade Features vs Target', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '08_grade_features.png')

print(df.groupby('target')[grade_cols].describe().round(2))
print()

# ===========================================================================
# Section 8: Derived Features
# ===========================================================================
print("=" * 60)
print("SECTION 8: Derived Features")
print("=" * 60)

derived_subset = [c for c in derived_cols if c != 'avg_attendance']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(derived_subset):
    for target_val, color, label in [
        (0, '#E74C3C', 'Tidak Tepat'), (1, '#2ECC71', 'Tepat Waktu')
    ]:
        data = df[df['target'] == target_val][col].dropna()
        axes[i].hist(data, bins=20, alpha=0.5, color=color, label=label)
    axes[i].set_title(f'{col} by Target', fontsize=11)
    axes[i].legend()

# Hide unused subplots
for j in range(len(derived_subset), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Derived Features vs Target', fontsize=14, y=1.02)
plt.tight_layout()
save_chart(fig, '09_derived_features.png')

print(df.groupby('target')[derived_subset].describe().round(3))
print()

# ===========================================================================
# Section 9: Attendance Analysis
# ===========================================================================
print("=" * 60)
print("SECTION 9: Attendance Analysis")
print("=" * 60)

att_data = df['avg_attendance'].dropna()
print(
    f'avg_attendance: {len(att_data)} non-null, '
    f'{df["avg_attendance"].isnull().sum()} missing '
    f'({df["avg_attendance"].isnull().mean()*100:.1f}%)'
)
print(f'Stats: mean={att_data.mean():.2f}, median={att_data.median():.2f}, std={att_data.std():.2f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Distribution
axes[0].hist(att_data, bins=20, alpha=0.7, color='#3498DB', edgecolor='white')
axes[0].axvline(
    att_data.median(), color='#E74C3C', linestyle='--',
    label=f'Median={att_data.median():.1f}'
)
axes[0].set_title('avg_attendance Distribution', fontsize=12)
axes[0].set_xlabel('%')
axes[0].legend()

# By target
for target_val, color, label in [
    (0, '#E74C3C', 'Tidak Tepat'), (1, '#2ECC71', 'Tepat Waktu')
]:
    data = df[df['target'] == target_val]['avg_attendance'].dropna()
    axes[1].hist(data, bins=20, alpha=0.5, color=color, label=label)
axes[1].set_title('avg_attendance by Target', fontsize=12)
axes[1].set_xlabel('%')
axes[1].legend()

# Missing target rate by has_attendance
df['has_attendance'] = df['avg_attendance'].notna().astype(int)
ct_att = pd.crosstab(df['has_attendance'], df['target'], normalize='index') * 100
ct_att.plot(kind='bar', ax=axes[2], color=['#E74C3C', '#2ECC71'])
axes[2].set_title('Target Rate: Has vs No Attendance Data', fontsize=12)
axes[2].set_xlabel('has_attendance (0=missing, 1=present)')
axes[2].set_ylabel('%')
axes[2].legend(['Tidak Tepat', 'Tepat Waktu'])
axes[2].set_xticklabels(['Missing', 'Present'], rotation=0)

plt.tight_layout()
save_chart(fig, '10_attendance_analysis.png')

print()
print('Target rate by has_attendance:')
print(ct_att.round(1))
print()

# ===========================================================================
# Section 10: Correlation Analysis
# ===========================================================================
print("=" * 60)
print("SECTION 10: Correlation Analysis")
print("=" * 60)

# Encode categoricals for correlation matrix
df_corr = df.copy()
df_corr['program_num'] = df_corr['program'].map({'AP': 0, 'IH': 1})
df_corr['gender_num'] = df_corr['jenis_kelamin'].map({'L': 0, 'P': 1})

corr_cols = (
    ['angkatan', 'program_num', 'gender_num', 'id_agama']
    + ips_cols + sks_cols + ['ipk_sem4', 'total_sks_lulus_sem4']
    + grade_cols + derived_cols + ['target']
)

corr_matrix = df_corr[corr_cols].corr()

# Chart 11: correlation heatmap
fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.6},
    annot_kws={'size': 8}
)
ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
save_chart(fig, '11_correlation_heatmap.png')

# Top correlations with target
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
print('Top correlations with target:')
print(target_corr)
print()

# Chart 12: horizontal bar chart of correlations with target
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in target_corr.values]
target_corr.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Correlation with Target', fontsize=14)
ax.set_xlabel('Pearson Correlation')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
save_chart(fig, '12_correlation_with_target.png')

# Chart 13: IPS inter-correlations heatmap
fig, ax = plt.subplots(figsize=(10, 8))
ips_corr = df_corr[ips_cols + ['ipk_sem4', 'avg_ips', 'ips_max', 'ips_min']].corr()
sns.heatmap(
    ips_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
    square=True, linewidths=0.5, annot_kws={'size': 9}
)
ax.set_title('IPS Feature Inter-Correlations', fontsize=14, pad=15)
plt.tight_layout()
save_chart(fig, '13_ips_intercorrelation.png')

# ===========================================================================
# Section 11: Semester Count & Target
# ===========================================================================
print("=" * 60)
print("SECTION 11: Semester Count & Target")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sem_ct = pd.crosstab(df['semester_count'], df['target'])
sem_ct.plot(kind='bar', stacked=True, ax=axes[0], color=['#E74C3C', '#2ECC71'])
axes[0].set_title('semester_count vs Target', fontsize=13)
axes[0].set_xlabel('Semester Count')
axes[0].legend(['Tidak Tepat', 'Tepat Waktu'])

sem_pct = pd.crosstab(df['semester_count'], df['target'], normalize='index') * 100
sem_pct[1].plot(kind='bar', ax=axes[1], color='#2ECC71')
axes[1].set_title('% Tepat Waktu by Semester Count', fontsize=13)
axes[1].set_ylabel('% Tepat Waktu')
axes[1].set_xlabel('Semester Count')
axes[1].axhline(88.8, color='#E74C3C', linestyle='--', label='Overall (88.8%)')
axes[1].legend()

plt.tight_layout()
save_chart(fig, '14_semester_count.png')

# ===========================================================================
# Section 12: Outlier Check (IPS)
# ===========================================================================
print("=" * 60)
print("SECTION 12: Outlier Check (IPS)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df[ips_cols].boxplot(ax=axes[0])
axes[0].set_title('IPS Boxplot (outlier detection)', fontsize=13)
axes[0].set_ylabel('IPS')

# IPS = 0.0 cases
zero_ips = df[(df[ips_cols] == 0.0).any(axis=1)]
print(f'Students with at least one IPS = 0.0: {len(zero_ips)}')
print(f'  Tepat waktu: {(zero_ips["target"]==1).sum()}')
print(f'  Tidak tepat: {(zero_ips["target"]==0).sum()}')

# IPS = 4.0 cases
perfect_ips = df[(df[ips_cols] == 4.0).any(axis=1)]
print(f'Students with at least one IPS = 4.0: {len(perfect_ips)}')

axes[1].scatter(
    df['angkatan'], df['avg_ips'], alpha=0.5,
    c=df['target'].map({0: '#E74C3C', 1: '#2ECC71'})
)
axes[1].set_title('avg_ips by Angkatan (colored by target)', fontsize=13)
axes[1].set_xlabel('Angkatan')
axes[1].set_ylabel('avg_ips')

plt.tight_layout()
save_chart(fig, '15_outlier_check.png')

print()

# ===========================================================================
# Section 13: Pairplot (Key Features)
# ===========================================================================
print("=" * 60)
print("SECTION 13: Pairplot (Key Features)")
print("=" * 60)

key_features = ['ips_sem1', 'ips_sem2', 'avg_ips', 'ipk_sem4',
                'failed_courses', 'ips_trend', 'target']
df_pair = df_eda[key_features].dropna()

pp = sns.pairplot(
    df_pair, hue='target', diag_kind='kde',
    palette={0: '#E74C3C', 1: '#2ECC71'}, height=2
)
pp.fig.suptitle('Pairplot of Key Features', y=1.02, fontsize=14)
save_chart(pp.fig, '16_pairplot.png')

print()
print("=" * 60)
print("EDA COMPLETE: All charts saved to eda-charts/")
print("=" * 60)
