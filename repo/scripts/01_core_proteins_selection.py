"""
Скрипт 1: Отбор коровых белков и базовая статистика
====================================================

ВХОД:  data/matrix_after_project_zscore_no_imputation.xlsx
ВЫХОД: results/tables/core_proteins_summary.csv
       results/figures/01_pca_visualization.png

Что делает:
  1. Загружает объединённую матрицу из 4 проектов PRIDE
  2. Размечает образцы по типу материала (Tissue/EVs)
  3. Отбирает коровые белки (coverage 100% во всех 180 образцах)
  4. Строит PCA-визуализацию по статусу и по проекту
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA


DATA_PATH = '../data/matrix_after_project_zscore_no_imputation.xlsx'
OUTPUT_FIGS = '../results/figures'
OUTPUT_TABLES = '../results/tables'
RANDOM_STATE = 42
EV_PROJECT = 'PXD009655'  

os.makedirs(OUTPUT_FIGS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)


df = pd.read_excel(DATA_PATH, sheet_name='matrix_after_project_z', engine='openpyxl')
ann = pd.read_excel(DATA_PATH, sheet_name='sample_annotation', engine='openpyxl')

META_COLS = ['Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs']
sample_cols = [c for c in df.columns if c not in META_COLS]

# Размечаем материал
ann['Material'] = ann['Project'].apply(lambda p: 'EVs' if p == EV_PROJECT else 'Tissue')

print(pd.crosstab(
    [ann['Material'], ann['Project']],
    ann['Status'],
    margins=True
).to_string())

data_full = df[sample_cols].astype(float)
coverage = (~data_full.isna()).sum(axis=1) / len(sample_cols)

print("  Распределение покрытия:")
for thr, label in [(1.00, '100% (CORE)'), (0.95, '≥95%'),
                   (0.90, '≥90%'), (0.80, '≥80%'), (0.50, '≥50%')]:
    n = (coverage >= thr).sum()
    print(f"    Coverage {label:>12s}: {n:>5d} белков")

core_mask = coverage >= 1.0
df_core = df[core_mask].reset_index(drop=True)
print(f"\n  Отобрано коровых белков: {len(df_core)}")

X = df_core[sample_cols].astype(float).T.values  # (180, n_proteins)
ann_idx = ann.set_index('SampleColumn').loc[sample_cols]
y = (ann_idx['Status'] == 'Cancer').astype(int).values
groups = ann_idx['Project'].values

pca = PCA(n_components=10, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Cum-10: {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%")

STATUS_COLORS = {'Cancer': '#E63946', 'Control': '#457B9D'}
PROJECT_COLORS = {'PXD033741': '#E63946', 'PXD009655': '#457B9D',
                  'PXD012998': '#2A9D8F', 'PXD025864': '#F4A261'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for status in ['Cancer', 'Control']:
    mask = ann_idx['Status'].values == status
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=STATUS_COLORS[status], label=f'{status} (n={mask.sum()})',
               alpha=0.75, s=60, edgecolors='white', linewidths=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA — по статусу', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[1]
for proj in sorted(np.unique(groups)):
    mask = groups == proj
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PROJECT_COLORS.get(proj, 'gray'),
               label=f'{proj} (n={mask.sum()})',
               alpha=0.75, s=60, edgecolors='white', linewidths=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA — по проекту (контроль batch)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

plt.suptitle(f'PCA на {len(df_core)} коровых белках', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/01_pca_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Сохранено: {OUTPUT_FIGS}/01_pca_visualization.png")

summary = pd.DataFrame({
    'Параметр': [
        'Образцов всего', 'Проектов', 'Cancer', 'Control',
        'Tissue образцов', 'EVs образцов',
        'Белков всего', 'Коровых белков (coverage 100%)',
        'Доля коровых, %',
    ],
    'Значение': [
        len(sample_cols), 4, int(y.sum()), int((y==0).sum()),
        int((ann_idx['Material']=='Tissue').sum()),
        int((ann_idx['Material']=='EVs').sum()),
        len(df), len(df_core),
        round(len(df_core)/len(df)*100, 1),
    ],
})
summary.to_csv(f'{OUTPUT_TABLES}/core_proteins_summary.csv', index=False)

