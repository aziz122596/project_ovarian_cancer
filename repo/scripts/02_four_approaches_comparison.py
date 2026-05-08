"""
Скрипт 2: Сравнение 4 подходов × 6 моделей машинного обучения
=============================================================

ВХОД:  data/matrix_after_project_zscore_no_imputation.xlsx
ВЫХОД: results/tables/summary_AUROC.csv
       results/tables/summary_AUPRC.csv
       results/tables/universal_markers.csv
       results/figures/02_summary_heatmap.png
       results/figures/03_universal_markers_scatter.png
       results/figures/04_roc_best_per_approach.png

Что делает:
  Сравнивает 4 стратегии интеграции мультицентровых данных:
    A) Tissue-only LOPO
    B) Universal markers (tissue ↔ EVs)
    C) Stratified — отдельные модели для tissue / EVs
    D) Material-aware feature engineering
  В каждом подходе тестирует 6 моделей: LogReg L1/L2, RF, GBM, SVM RBF/Linear.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    accuracy_score, f1_score
)

DATA_PATH = '../data/matrix_after_project_zscore_no_imputation.xlsx'
OUTPUT_FIGS = '../results/figures'
OUTPUT_TABLES = '../results/tables'
RANDOM_STATE = 42
EV_PROJECT = 'PXD009655'
N_SPLITS_KFOLD_EVS = 5

UNIVERSAL_P_THRESHOLD = 0.05
UNIVERSAL_T_THRESHOLD = 1.5

os.makedirs(OUTPUT_FIGS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name='matrix_after_project_z', engine='openpyxl')
ann = pd.read_excel(DATA_PATH, sheet_name='sample_annotation', engine='openpyxl')

META_COLS = ['Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs']
sample_cols = [c for c in df.columns if c not in META_COLS]
ann['Material'] = ann['Project'].apply(lambda p: 'EVs' if p == EV_PROJECT else 'Tissue')

data_full = df[sample_cols].astype(float)
core_mask = (~data_full.isna()).all(axis=1)
df_core = df[core_mask].reset_index(drop=True)
feat_names = [
    str(r['Gene names']).strip() if pd.notna(r['Gene names']) and str(r['Gene names']).strip()
    else str(r['Protein IDs'])
    for _, r in df_core.iterrows()
]


X_full = df_core[sample_cols].astype(float).T.values
ann_idx = ann.set_index('SampleColumn').loc[sample_cols]
y_full = (ann_idx['Status'] == 'Cancer').astype(int).values
groups_full = ann_idx['Project'].values
material_full = ann_idx['Material'].values

tissue_mask = material_full == 'Tissue'
ev_mask = material_full == 'EVs'
X_T = X_full[tissue_mask]; y_T = y_full[tissue_mask]; g_T = groups_full[tissue_mask]
X_E = X_full[ev_mask]; y_E = y_full[ev_mask]
print(f"  Tissue: {X_T.shape}   EVs: {X_E.shape}")

def make_models():
    return {
        'LogReg L2': LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
                                         max_iter=2000, solver='lbfgs',
                                         random_state=RANDOM_STATE),
        'LogReg L1': LogisticRegression(penalty='l1', C=1.0, class_weight='balanced',
                                         max_iter=2000, solver='liblinear',
                                         random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                 max_features='sqrt',
                                                 random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                         max_depth=3, random_state=RANDOM_STATE),
        'SVM RBF': SVC(kernel='rbf', C=1.0, class_weight='balanced',
                       probability=True, random_state=RANDOM_STATE),
        'SVM Linear': SVC(kernel='linear', C=1.0, class_weight='balanced',
                          probability=True, random_state=RANDOM_STATE),
    }


def get_proba(m, Xt):
    if hasattr(m, 'predict_proba'):
        return m.predict_proba(Xt)[:, 1]
    return m.decision_function(Xt)


def eval_split(X_tr, y_tr, X_te, y_te, base):
    m = base.__class__(**base.get_params())
    m.fit(X_tr, y_tr)
    proba = get_proba(m, X_te)
    pred = m.predict(X_te)
    rec = {
        'n_test': len(y_te),
        'acc': accuracy_score(y_te, pred),
        'f1': f1_score(y_te, pred, zero_division=0),
        'proba': proba, 'y_true': y_te,
    }
    if len(np.unique(y_te)) == 2:
        rec['auroc'] = roc_auc_score(y_te, proba)
        rec['auprc'] = average_precision_score(y_te, proba)
    else:
        rec['auroc'] = np.nan
        rec['auprc'] = np.nan
    return rec


def run_lopo(X, y, groups, label):
    print(f"\n── {label} ──")
    logo = LeaveOneGroupOut()
    results = {}
    for name, base in make_models().items():
        pooled_yt, pooled_yp = [], []
        for tr_idx, te_idx in logo.split(X, y, groups):
            rec = eval_split(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], base)
            pooled_yt.extend(rec['y_true']); pooled_yp.extend(rec['proba'])
        pyt, pyp = np.array(pooled_yt), np.array(pooled_yp)
        results[name] = {
            'pooled_auroc': roc_auc_score(pyt, pyp),
            'pooled_auprc': average_precision_score(pyt, pyp),
            'y_true': pyt, 'y_proba': pyp,
        }
        print(f"  {name:20s} AUROC={results[name]['pooled_auroc']:.3f}  "
              f"AUPRC={results[name]['pooled_auprc']:.3f}")
    return results


A_results = run_lopo(X_T, y_T, g_T, "A: 3 проекта ткани")


def t_stats(X, y):
    ts = np.zeros(X.shape[1]); ps = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        a, b = X[y == 1, i], X[y == 0, i]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            ts[i], ps[i] = t, p
    return ts, ps

t_T_arr, p_T_arr = t_stats(X_T, y_T)
t_E_arr, p_E_arr = t_stats(X_E, y_E)

valid = (~np.isnan(t_T_arr)) & (~np.isnan(t_E_arr))
spearman_corr, spearman_p = stats.spearmanr(t_T_arr[valid], t_E_arr[valid])
print(f"\n  Spearman ρ (tissue vs EVs t-stats): {spearman_corr:+.3f}, p={spearman_p:.2e}")

universal_mask = (
    (p_T_arr < UNIVERSAL_P_THRESHOLD) & (p_E_arr < UNIVERSAL_P_THRESHOLD) &
    (np.sign(t_T_arr) == np.sign(t_E_arr)) &
    (np.abs(t_T_arr) > UNIVERSAL_T_THRESHOLD) & (np.abs(t_E_arr) > UNIVERSAL_T_THRESHOLD)
)
n_universal = int(universal_mask.sum())
print(f"  Universal markers: {n_universal}")

universal_df = pd.DataFrame({
    'Gene': feat_names,
    't_tissue': t_T_arr, 'p_tissue': p_T_arr,
    't_EVs': t_E_arr, 'p_EVs': p_E_arr,
    'is_universal': universal_mask,
})
universal_df.to_csv(f'{OUTPUT_TABLES}/universal_markers.csv', index=False)

if n_universal < 20:
    soft_score = (-np.log10(np.maximum(p_T_arr, 1e-300))
                  - np.log10(np.maximum(p_E_arr, 1e-300))) * (np.sign(t_T_arr) == np.sign(t_E_arr))
    universal_idx = np.argsort(-soft_score)[:50]
else:
    universal_idx = np.where(universal_mask)[0]
print(f"  Используем {len(universal_idx)} белков для модели B")

X_universal = X_full[:, universal_idx]
B_results = run_lopo(X_universal, y_full, groups_full, "B: LOPO на universal")

print("\n" + "=" * 65)
print("C) STRATIFIED — EVs internal CV")
print("=" * 65)
print("  C-1: Tissue model = идентично результатам A")

skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD_EVS, shuffle=True, random_state=RANDOM_STATE)
C_ev_results = {}
for name, base in make_models().items():
    pooled_yt, pooled_yp = [], []
    fold_aurocs = []
    for tr, te in skf.split(X_E, y_E):
        if len(np.unique(y_E[te])) < 2:
            continue
        rec = eval_split(X_E[tr], y_E[tr], X_E[te], y_E[te], base)
        if not np.isnan(rec['auroc']):
            fold_aurocs.append(rec['auroc'])
        pooled_yt.extend(rec['y_true']); pooled_yp.extend(rec['proba'])
    pyt, pyp = np.array(pooled_yt), np.array(pooled_yp)
    C_ev_results[name] = {
        'pooled_auroc': roc_auc_score(pyt, pyp),
        'pooled_auprc': average_precision_score(pyt, pyp),
        'y_true': pyt, 'y_proba': pyp,
    }
    print(f"  {name:20s} pooled AUROC={C_ev_results[name]['pooled_auroc']:.3f}")

X_d = X_full.copy()
for material_label in ['Tissue', 'EVs']:
    mat_mask = material_full == material_label
    if mat_mask.sum() > 1:
        X_d[mat_mask] = StandardScaler().fit_transform(X_d[mat_mask])
material_feat = (material_full == 'EVs').astype(float).reshape(-1, 1)
X_d_aug = np.hstack([X_d, material_feat])
D_results = run_lopo(X_d_aug, y_full, groups_full, "D: material-aware LOPO")

models = list(make_models().keys())
summary_auroc = pd.DataFrame({
    'A: Tissue-only LOPO':       [A_results[m]['pooled_auroc'] for m in models],
    'B: Universal markers LOPO': [B_results[m]['pooled_auroc'] for m in models],
    'C: EVs internal CV':        [C_ev_results[m]['pooled_auroc'] for m in models],
    'D: Material-aware LOPO':    [D_results[m]['pooled_auroc'] for m in models],
}, index=models)
summary_auroc.to_csv(f'{OUTPUT_TABLES}/summary_AUROC.csv')


summary_auprc = pd.DataFrame({
    'A: Tissue-only LOPO':       [A_results[m]['pooled_auprc'] for m in models],
    'B: Universal markers LOPO': [B_results[m]['pooled_auprc'] for m in models],
    'C: EVs internal CV':        [C_ev_results[m]['pooled_auprc'] for m in models],
    'D: Material-aware LOPO':    [D_results[m]['pooled_auprc'] for m in models],
}, index=models)
summary_auprc.to_csv(f'{OUTPUT_TABLES}/summary_AUPRC.csv')

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(summary_auroc, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.5, vmax=1.0, linewidths=0.4,
            cbar_kws={'label': 'pooled AUROC'}, ax=ax)
ax.set_title('AUROC по 4 подходам × 6 моделей', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/02_summary_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(t_T_arr, t_E_arr, alpha=0.3, s=8, color='gray',
           label=f'Все белки (n={len(t_T_arr)})')
ax.scatter(t_T_arr[universal_mask], t_E_arr[universal_mask], alpha=0.85, s=30,
           color='#E63946', label=f'Universal (n={n_universal})')
ax.axhline(0, color='black', lw=0.6, alpha=0.5)
ax.axvline(0, color='black', lw=0.6, alpha=0.5)
ax.set_xlabel('t-statistic Cancer vs Control — Tissue')
ax.set_ylabel('t-statistic Cancer vs Control — EVs')
ax.set_title(f'Сонаправленность сигналов\nSpearman ρ = {spearman_corr:+.3f}',
             fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/03_universal_markers_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC 
fig, ax = plt.subplots(figsize=(8, 7))
best_A = summary_auroc['A: Tissue-only LOPO'].idxmax()
best_B = summary_auroc['B: Universal markers LOPO'].idxmax()
best_C = summary_auroc['C: EVs internal CV'].idxmax()
best_D = summary_auroc['D: Material-aware LOPO'].idxmax()
for label, res, color in [
    (f'A: Tissue-only ({best_A})', A_results[best_A], '#2A9D8F'),
    (f'B: Universal ({best_B})', B_results[best_B], '#E76F51'),
    (f'C: EVs ({best_C})', C_ev_results[best_C], '#457B9D'),
    (f'D: Material-aware ({best_D})', D_results[best_D], '#F4A261'),
]:
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
    auc = roc_auc_score(res['y_true'], res['y_proba'])
    ax.plot(fpr, tpr, lw=2.2, color=color, label=f'{label}, AUC={auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC-кривые лучших моделей в каждом подходе', fontweight='bold')
ax.legend(loc='lower right', fontsize=10); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/04_roc_best_per_approach.png', dpi=150, bbox_inches='tight')
plt.close()
