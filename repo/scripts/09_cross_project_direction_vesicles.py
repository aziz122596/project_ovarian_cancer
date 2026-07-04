"""
Скрипт 9: Межпроектная корреляция пониженные белки вклад везикул

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

DATA_PATH   = '../data/matrix_after_project_zscore_no_imputation.xlsx'
UNI_PATH    = '../results/tables/universal_markers.csv'
TAB         = '../results/tables'
FIG         = '../results/figures'
P_THR, T_THR = 0.05, 1.5
TISSUE_PROJECTS = ['PXD033741', 'PXD012998', 'PXD025864']
EV_PROJECTS     = ['PXD009655']
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def t_stats(X, y):
    """Welch t по каждому белку: рак(1) против контроля(0). Возвращает t, p."""
    t = np.full(X.shape[1], np.nan); p = np.ones(X.shape[1])
    a_idx, b_idx = y == 1, y == 0
    for i in range(X.shape[1]):
        a, b = X[a_idx, i], X[b_idx, i]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            t[i], p[i] = stats.ttest_ind(a, b, equal_var=False)
    return t, p

print("Загрузка матрицы…")
mat = pd.read_excel(DATA_PATH, sheet_name='matrix_after_project_z', engine='openpyxl')
ann = pd.read_excel(DATA_PATH, sheet_name='sample_annotation', engine='openpyxl')
id_cols = ['Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs']
sample_cols = [c for c in mat.columns if c not in id_cols]
genes = mat['Gene names'].astype(str).values
ann = ann.set_index('SampleColumn').loc[sample_cols]

X_all = mat[sample_cols].astype(float).T.values   
y_all = (ann['Status'].values == 'Cancer').astype(int)
proj_all = ann['Project'].values

core_mask = ~np.isnan(X_all).any(axis=0)
X_all = X_all[:, core_mask]; genes = genes[core_mask]
print(f"  {X_all.shape[0]} образцов × {X_all.shape[1]} коровых белков; проекты: {sorted(set(proj_all))}")

print("\n[3] Корреляция между проектами (t рак/контроль)…")
proj_order = TISSUE_PROJECTS + EV_PROJECTS
tmat = {}
for pj in proj_order:
    m = proj_all == pj
    if len(np.unique(y_all[m])) < 2:
        continue
    t, _ = t_stats(X_all[m], y_all[m])
    tmat[pj] = t
Tdf = pd.DataFrame(tmat, index=genes)

sp = Tdf.corr(method='spearman')
pe = Tdf.corr(method='pearson')
sp.to_csv(f'{TAB}/09_project_correlation_spearman.csv')
pe.to_csv(f'{TAB}/09_project_correlation_pearson.csv')
print("  Spearman ρ между проектами:")
print(sp.round(3).to_string())

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(sp.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp.columns, rotation=45, ha='right')
ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index)
for i in range(len(sp)):
    for j in range(len(sp)):
        ax.text(j, i, f"{sp.values[i, j]:.2f}", ha='center', va='center',
                color='white' if abs(sp.values[i, j]) > 0.5 else 'black', fontsize=9)
ax.set_title('Корреляция эффектов рак/контроль между проектами (Spearman)')
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f'{FIG}/09_project_correlation.png', dpi=200); plt.close(fig)

tt = [sp.loc[a, b] for i, a in enumerate(TISSUE_PROJECTS) for b in TISSUE_PROJECTS[i+1:]]
te = [sp.loc[a, EV_PROJECTS[0]] for a in TISSUE_PROJECTS if EV_PROJECTS[0] in sp.columns]
print(f"  Средняя ρ ткань–ткань: {np.mean(tt):.3f}")
if te:
    print(f"  Средняя ρ ткань–везикулы: {np.mean(te):.3f}")


print("\n[2] Направление универсальных маркеров…")
uni = pd.read_csv(UNI_PATH)
U = uni[uni['is_universal']].copy()
U['direction'] = np.where(U['t_tissue'] > 0, 'UP', 'DOWN')
up = U[U['direction'] == 'UP']; down = U[U['direction'] == 'DOWN']
print(f"  Универсальных: {len(U)}  |  UP: {len(up)} ({len(up)/len(U)*100:.1f}%)  |  DOWN: {len(down)}")
U.to_csv(f'{TAB}/09_universal_with_direction.csv', index=False)
down.sort_values('t_tissue').to_csv(f'{TAB}/09_universal_DOWN.csv', index=False)
print(f"  Пониженные универсальные маркеры: {', '.join(down['Gene'].tolist())}")

print("  RF importance (ткань) → пересечение с DOWN…")
tis_mask = np.isin(proj_all, TISSUE_PROJECTS)
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                            max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_all[tis_mask], y_all[tis_mask])
imp = pd.Series(rf.feature_importances_, index=genes).sort_values(ascending=False)
top100 = set(imp.head(100).index)
down_genes = set(down['Gene'])
important_down = sorted(top100 & down_genes)

uni_dedup = uni.drop_duplicates(subset='Gene').set_index('Gene')
top100_genes = list(imp.head(100).index)
top100_dir = uni_dedup.reindex([g for g in top100_genes if g in uni_dedup.index])
top100_down_any = top100_dir[top100_dir['t_tissue'] < 0]
pd.DataFrame({'Gene': list(important_down)}).to_csv(f'{TAB}/09_important_but_DOWN_universal.csv', index=False)
top100_down_any.reset_index().to_csv(f'{TAB}/09_top100RF_downregulated.csv', index=False)
print(f"  Важные (top-100 RF) и пониженные (любые): {len(top100_down_any)}")
print(f"    среди них универсальные DOWN: {important_down if important_down else '—'}")

print("\n[4] Вклад везикул…")
u = uni.copy()
sig_T = (u['p_tissue'] < P_THR) & (u['t_tissue'].abs() > T_THR)
sig_E = (u['p_EVs']   < P_THR) & (u['t_EVs'].abs()   > T_THR)
same_dir = np.sign(u['t_tissue']) == np.sign(u['t_EVs'])

lost   = u[sig_T & ~sig_E]                    
gained = u[~sig_T & sig_E]                    
both_same = u[sig_T & sig_E & same_dir]        
flip   = u[sig_T & sig_E & ~same_dir]          

lost.to_csv(f'{TAB}/09_vesicles_LOST.csv', index=False)
gained.to_csv(f'{TAB}/09_vesicles_GAINED.csv', index=False)
flip.to_csv(f'{TAB}/09_vesicles_DIRECTION_FLIP.csv', index=False)


print("\n[1] Task A \\ Task B…")
taskA_list = f'{TAB}/taskA_top_proteins.csv'   #
if os.path.exists(taskA_list):
    A = pd.read_csv(taskA_list)
    taskB = set(genes[np.array([g in set(uni[uni['is_universal']]['Gene']) for g in genes])])
   
    A['in_taskB'] = A['Gene'].isin(taskB)
    A_specific = A[~A['in_taskB']]
    A_specific.to_csv(f'{TAB}/09_taskA_specific.csv', index=False)
