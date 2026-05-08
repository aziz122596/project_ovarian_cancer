"""
Скрипт 3: Расширенная валидация Tissue-only моделей
====================================================

ВХОД:  data/matrix_after_project_zscore_no_imputation.xlsx
ВЫХОД: results/tables/extended_validation_metrics.csv
       results/tables/hallmark_enrichment_top100.csv
       results/tables/hallmark_enrichment_universal.csv
       results/figures/05_bootstrap.png
       results/figures/06_permutation.png
       results/figures/07_roc_with_threshold.png
       results/figures/08_confusion.png
       results/figures/09_stability.png
       results/figures/10_enrichment.png

Что делает:
  1. Bootstrap CI (1000 перевыборок) для AUROC и AUPRC
  2. Permutation test (100 перестановок меток) для проверки значимости
  3. Расчёт оптимального порога по Юдену + клинические метрики
  4. Тест устойчивости при удалении доминирующего проекта PXD033741
  5. Hallmark enrichment топ-100 RF и universal markers

ВНИМАНИЕ: Permutation test занимает несколько минут. Для ускорения
используется уменьшенный Random Forest (n_estimators=100).
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, accuracy_score
)

DATA_PATH = '../data/matrix_after_project_zscore_no_imputation.xlsx'
OUTPUT_FIGS = '../results/figures'
OUTPUT_TABLES = '../results/tables'
RANDOM_STATE = 42
EV_PROJECT = 'PXD009655'
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 100
N_FEATURES_PERM = 200    # для ускорения permutation test

os.makedirs(OUTPUT_FIGS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name='matrix_after_project_z', engine='openpyxl')
ann = pd.read_excel(DATA_PATH, sheet_name='sample_annotation', engine='openpyxl')
META = ['Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs']
sample_cols = [c for c in df.columns if c not in META]
ann['Material'] = ann['Project'].apply(lambda p: 'EVs' if p == EV_PROJECT else 'Tissue')

data = df[sample_cols].astype(float)
core = (~data.isna()).all(axis=1)
df_core = df[core].reset_index(drop=True)
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
print(f"Tissue: {X_T.shape}")


def get_proba(m, Xt):
    return m.predict_proba(Xt)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(Xt)


def lopo_predictions(X, y, groups, model_factory):
    logo = LeaveOneGroupOut()
    yt, yp = [], []
    for tr, te in logo.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        yp.extend(get_proba(m, X[te]))
        yt.extend(y[te])
    return np.array(yt), np.array(yp)


def lopo_with_fs(X, y, groups, model_fn, k):
    """LOPO с feature selection внутри каждого fold (для ускорения permutation test)."""
    logo = LeaveOneGroupOut()
    yt, yp = [], []
    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if k is not None and k < X.shape[1]:
            sel = SelectKBest(f_classif, k=k).fit(Xtr, y[tr])
            Xtr = sel.transform(Xtr)
            Xte = sel.transform(Xte)
        m = model_fn()
        m.fit(Xtr, y[tr])
        yp.extend(get_proba(m, Xte))
        yt.extend(y[te])
    return np.array(yt), np.array(yp)


def rf_factory():
    return RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                   max_features='sqrt',
                                   random_state=RANDOM_STATE, n_jobs=-1)


def rf_factory_fast():
    return RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   max_features='sqrt',
                                   random_state=RANDOM_STATE, n_jobs=-1)


def lr_l2_factory():
    return LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
                               max_iter=2000, solver='lbfgs',
                               random_state=RANDOM_STATE)

y_true_RF, y_proba_RF = lopo_predictions(X_T, y_T, g_T, rf_factory)
y_true_LR, y_proba_LR = lopo_predictions(X_T, y_T, g_T, lr_l2_factory)
obs_RF = roc_auc_score(y_true_RF, y_proba_RF)
obs_LR = roc_auc_score(y_true_LR, y_proba_LR)

def bootstrap(y_true, y_proba, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_proba[idx]))
    return np.array(aurocs), np.array(auprcs)

br_RF, bp_RF = bootstrap(y_true_RF, y_proba_RF, n=N_BOOTSTRAP, seed=42)
br_LR, bp_LR = bootstrap(y_true_LR, y_proba_LR, n=N_BOOTSTRAP, seed=43)
ci_RF_auroc = np.percentile(br_RF, [2.5, 97.5])
ci_LR_auroc = np.percentile(br_LR, [2.5, 97.5])
ci_RF_auprc = np.percentile(bp_RF, [2.5, 97.5])
ci_LR_auprc = np.percentile(bp_LR, [2.5, 97.5])

yt_RF, yp_RF = lopo_with_fs(X_T, y_T, g_T, rf_factory_fast, k=N_FEATURES_PERM)
yt_LR, yp_LR = lopo_with_fs(X_T, y_T, g_T, lr_l2_factory, k=N_FEATURES_PERM)
obs_RF_perm = roc_auc_score(yt_RF, yp_RF)
obs_LR_perm = roc_auc_score(yt_LR, yp_LR)

rng = np.random.default_rng(42)
null_RF, null_LR = [], []
t0 = time.time()
for i in range(N_PERMUTATIONS):
    y_perm = rng.permutation(y_T)
    try:
        yt1, yp1 = lopo_with_fs(X_T, y_perm, g_T, rf_factory_fast, k=N_FEATURES_PERM)
        yt2, yp2 = lopo_with_fs(X_T, y_perm, g_T, lr_l2_factory, k=N_FEATURES_PERM)
        null_RF.append(roc_auc_score(yt1, yp1))
        null_LR.append(roc_auc_score(yt2, yp2))
    except Exception:
        continue
    if (i+1) % 10 == 0:
        elapsed = time.time() - t0

null_RF = np.array(null_RF); null_LR = np.array(null_LR)
p_RF = (np.sum(null_RF >= obs_RF_perm) + 1) / (len(null_RF) + 1)
p_LR = (np.sum(null_LR >= obs_LR_perm) + 1) / (len(null_LR) + 1)

def clinical(yt, yp):
    fpr, tpr, thr = roc_curve(yt, yp)
    j = tpr - fpr
    bi = np.argmax(j)
    bt = thr[bi]
    yp_pred = (yp >= bt).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp_pred).ravel()
    return {
        'threshold': float(bt),
        'sensitivity': tp/(tp+fn) if tp+fn else 0,
        'specificity': tn/(tn+fp) if tn+fp else 0,
        'ppv': tp/(tp+fp) if tp+fp else 0,
        'npv': tn/(tn+fn) if tn+fn else 0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'fpr': fpr, 'tpr': tpr,
    }

cl_RF = clinical(y_true_RF, y_proba_RF)
cl_LR = clinical(y_true_LR, y_proba_LR)
print(f"  RF: thr={cl_RF['threshold']:.3f}  Sens={cl_RF['sensitivity']:.3f}  Spec={cl_RF['specificity']:.3f}")
print(f"  LR: thr={cl_LR['threshold']:.3f}  Sens={cl_LR['sensitivity']:.3f}  Spec={cl_LR['specificity']:.3f}")

print("\n── Stability test ──")
small_mask = np.isin(g_T, ['PXD012998', 'PXD025864'])
X_T_s = X_T[small_mask]; y_T_s = y_T[small_mask]; g_T_s = g_T[small_mask]
yt_s_RF, yp_s_RF = lopo_predictions(X_T_s, y_T_s, g_T_s, rf_factory)
yt_s_LR, yp_s_LR = lopo_predictions(X_T_s, y_T_s, g_T_s, lr_l2_factory)
auc_RF_s = roc_auc_score(yt_s_RF, yp_s_RF)
auc_LR_s = roc_auc_score(yt_s_LR, yp_s_LR)

big_test = g_T == 'PXD033741'
small_train = ~big_test
m_RF = rf_factory(); m_RF.fit(X_T[small_train], y_T[small_train])
auc_alt_RF = roc_auc_score(y_T[big_test], get_proba(m_RF, X_T[big_test]))
m_LR = lr_l2_factory(); m_LR.fit(X_T[small_train], y_T[small_train])
auc_alt_LR = roc_auc_score(y_T[big_test], get_proba(m_LR, X_T[big_test]))
print(f"  Без PXD033741: RF={auc_RF_s:.4f}, LR={auc_LR_s:.4f}")
print(f"  Train(small) → Test(big): RF={auc_alt_RF:.4f}, LR={auc_alt_LR:.4f}")

print("\n── Hallmark enrichment ──")
rf_full = rf_factory(); rf_full.fit(X_T, y_T)
imp = pd.Series(rf_full.feature_importances_, index=feat_names).sort_values(ascending=False)
top100 = set(imp.head(100).index.tolist())

def t_stats_local(X, y):
    ts = np.zeros(X.shape[1]); ps = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        a, b = X[y == 1, i], X[y == 0, i]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            ts[i], ps[i] = t, p
    return ts, ps

t_T_arr, p_T_arr = t_stats_local(X_T, y_T)
t_E_arr, p_E_arr = t_stats_local(X_E, y_E)
universal_idx = np.where(
    (p_T_arr < 0.05) & (p_E_arr < 0.05) &
    (np.sign(t_T_arr) == np.sign(t_E_arr)) &
    (np.abs(t_T_arr) > 1.5) & (np.abs(t_E_arr) > 1.5)
)[0]
universal_genes = set([feat_names[i] for i in universal_idx])

# hallmark-категории (выжимка из MSigDB / KEGG / Reactome)
hallmark_sets = {
    'Glycolysis_Warburg': {'HK1','HK2','PFKL','PFKM','PFKP','GAPDH','PKM','LDHA','LDHB','ENO1','ENO2','PGK1','TPI1','ALDOA','ALDOC','GPI','PGAM1','BPGM'},
    'OXPHOS_Mitochondrial': {'ATP5F1A','ATP5F1B','ATP5J','ATP5J2','ATP5MC1','ATP5MC2','ATP5PD','COX4I1','COX5A','COX5B','COX6A1','COX6B1','COX7A2','COX7C','COX8A','NDUFA1','NDUFA2','NDUFB1','NDUFS1','NDUFV1','UQCRC1','UQCRC2','UQCRH','CYC1','IDH2','IDH3A','IDH3B','IDH3G','SDHA','SDHB','OGDH','CS','MDH2','AIFM1'},
    'TCA_cycle': {'CS','ACO2','IDH2','IDH3A','IDH3B','IDH3G','OGDH','SUCLA2','SUCLG1','SDHA','SDHB','FH','MDH2'},
    'Translation_Ribosome': {'EIF2S1','EIF2S2','EIF2S3','EIF3A','EIF3B','EIF3D','EIF3E','EIF3F','EIF3G','EIF3H','EIF3I','EIF3L','EIF4A1','EIF4A2','EIF4B','EIF4G1','EIF4E','EEF1A1','EEF1A2','EEF1G','EEF2','RPL3','RPL4','RPL5','RPL6','RPL7','RPL10','RPL11','RPL13','RPL14','RPL15','RPL18','RPL35A','RPS3','RPS4X','RPS5','RPS6','RPS8','RPS14'},
    'Apoptosis': {'BAX','BCL2','BID','CASP3','CASP6','CASP7','CASP8','CASP9','PARP1','AIFM1','CYCS','DIABLO','XIAP','BIRC5','TP53','BAD','BBC3','PMAIP1','BAK1'},
    'Cell_cycle_Proliferation': {'MCM2','MCM3','MCM4','MCM5','MCM6','MCM7','CDK1','CDK2','CDK4','CDK6','CCNA2','CCNB1','CCNB2','CCND1','CCNE1','PCNA','MKI67','TOP2A','AURKA','AURKB','PLK1','BUB1','BUB1B','CDC20','BIRC5'},
    'DNA_damage_repair': {'BRCA1','BRCA2','ATM','ATR','RAD51','XRCC1','XRCC4','XRCC5','XRCC6','PARP1','MSH2','MSH6','MLH1','PMS2','PALB2','CHEK1','CHEK2','H2AFX','MDC1','TP53BP1'},
    'EMT_Cytoskeleton': {'VIM','CDH1','CDH2','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2','FN1','ACTA2','TAGLN','CALD1','CNN1','TPM1','TPM2','MYL9','MYH11','JUP'},
    'Immune_Inflammation': {'TNF','IL6','IL1B','IFNG','NFKB1','STAT1','STAT3','IRF1','IRF7','MX1','OAS1','ISG15','IFIT1','IFIT3','CXCL10','CXCL11','B2M','HLA-A','HLA-B','HLA-C'},
    'Protein_folding_ER': {'HSP90AA1','HSP90AB1','HSPA1A','HSPA5','HSPA8','HSPB1','CALR','CANX','PDIA3','PDIA4','PDIA6','P4HB','SEC11A','SEC23IP','SEC61B','SSR4','TMCO1','COPB1','COPB2','COPA','COPZ1','ARF5'},
}

universe = set(feat_names)
def hyper(query, hallmark, univ):
    q = query & univ; h = hallmark & univ; ov = q & h
    M, n, N, k = len(univ), len(h), len(q), len(ov)
    if M == 0 or n == 0 or N == 0:
        return None
    return {
        'overlap': k, 'expected': N*n/M,
        'fold_enrichment': (k/N)/(n/M) if N and n else 0,
        'pvalue': stats.hypergeom.sf(k-1, M, n, N),
        'genes': sorted(ov),
    }

rows100, rows_uni = [], []
for hn, gs in hallmark_sets.items():
    r1 = hyper(top100, gs, universe)
    r2 = hyper(universal_genes, gs, universe)
    if r1:
        rows100.append({'Hallmark': hn,
                        'Top100_overlap': r1['overlap'],
                        'Expected': round(r1['expected'], 2),
                        'Fold_enrichment': round(r1['fold_enrichment'], 2),
                        'p_value': r1['pvalue'],
                        'Genes': ', '.join(r1['genes'][:8])})
    if r2:
        rows_uni.append({'Hallmark': hn,
                         'Universal_overlap': r2['overlap'],
                         'Expected': round(r2['expected'], 2),
                         'Fold_enrichment': round(r2['fold_enrichment'], 2),
                         'p_value': r2['pvalue'],
                         'Genes': ', '.join(r2['genes'][:8])})

e1 = pd.DataFrame(rows100).sort_values('p_value')
e2 = pd.DataFrame(rows_uni).sort_values('p_value')
e1['p_adj_BH'] = stats.false_discovery_control(e1['p_value'].values)
e2['p_adj_BH'] = stats.false_discovery_control(e2['p_value'].values)
e1.to_csv(f'{OUTPUT_TABLES}/hallmark_enrichment_top100.csv', index=False)
e2.to_csv(f'{OUTPUT_TABLES}/hallmark_enrichment_universal.csv', index=False)

metrics_df = pd.DataFrame({
    'Метрика': [
        'AUROC (pooled LOPO)',
        '95% CI AUROC',
        'AUPRC (pooled LOPO)',
        '95% CI AUPRC',
        'Permutation p-value',
        'Sensitivity (Youden)',
        'Specificity',
        'PPV',
        'NPV',
        'Optimal threshold',
        'AUROC без PXD033741',
        'AUROC train(small)→test(big)',
    ],
    'Random Forest': [
        f'{obs_RF:.4f}',
        f'[{ci_RF_auroc[0]:.4f}, {ci_RF_auroc[1]:.4f}]',
        f'{average_precision_score(y_true_RF, y_proba_RF):.4f}',
        f'[{ci_RF_auprc[0]:.4f}, {ci_RF_auprc[1]:.4f}]',
        f'{p_RF:.4f}',
        f'{cl_RF["sensitivity"]:.3f}',
        f'{cl_RF["specificity"]:.3f}',
        f'{cl_RF["ppv"]:.3f}',
        f'{cl_RF["npv"]:.3f}',
        f'{cl_RF["threshold"]:.3f}',
        f'{auc_RF_s:.4f}',
        f'{auc_alt_RF:.4f}',
    ],
    'Logistic L2': [
        f'{obs_LR:.4f}',
        f'[{ci_LR_auroc[0]:.4f}, {ci_LR_auroc[1]:.4f}]',
        f'{average_precision_score(y_true_LR, y_proba_LR):.4f}',
        f'[{ci_LR_auprc[0]:.4f}, {ci_LR_auprc[1]:.4f}]',
        f'{p_LR:.4f}',
        f'{cl_LR["sensitivity"]:.3f}',
        f'{cl_LR["specificity"]:.3f}',
        f'{cl_LR["ppv"]:.3f}',
        f'{cl_LR["npv"]:.3f}',
        f'{cl_LR["threshold"]:.3f}',
        f'{auc_LR_s:.4f}',
        f'{auc_alt_LR:.4f}',
    ],
})
metrics_df.to_csv(f'{OUTPUT_TABLES}/extended_validation_metrics.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, br, obs, ci, name, color in [
    (axes[0], br_RF, obs_RF, ci_RF_auroc, 'Random Forest', 'seagreen'),
    (axes[1], br_LR, obs_LR, ci_LR_auroc, 'Logistic L2', 'steelblue'),
]:
    ax.hist(br, bins=50, color=color, alpha=0.65, edgecolor='white')
    ax.axvline(obs, color='red', lw=2.2, label=f'Observed = {obs:.3f}')
    ax.axvline(ci[0], color='black', ls='--', alpha=0.7,
               label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    ax.axvline(ci[1], color='black', ls='--', alpha=0.7)
    ax.set_xlabel('AUROC'); ax.set_ylabel('Frequency')
    ax.set_title(f'Bootstrap distribution\n{name}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/05_bootstrap.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, null, obs, p, name in [
    (axes[0], null_RF, obs_RF_perm, p_RF, 'Random Forest'),
    (axes[1], null_LR, obs_LR_perm, p_LR, 'Logistic L2'),
]:
    ax.hist(null, bins=30, color='lightgray', edgecolor='gray', alpha=0.85,
            label=f'Null (n={len(null)})')
    ax.axvline(obs, color='red', lw=2.5, label=f'Observed = {obs:.3f}\np = {p:.4f}')
    ax.axvline(0.5, color='black', ls=':', alpha=0.5, label='Random')
    ax.set_xlim(0.3, 1.0); ax.set_xlabel('AUROC'); ax.set_ylabel('Frequency')
    ax.set_title(f'Permutation test\n{name}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/06_permutation.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC + threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, cl, name, obs in [
    (axes[0], cl_RF, 'Random Forest', obs_RF),
    (axes[1], cl_LR, 'Logistic L2', obs_LR),
]:
    fpr = np.array(cl['fpr']); tpr = np.array(cl['tpr'])
    ax.plot(fpr, tpr, lw=2.2, color='steelblue', label=f'AUROC = {obs:.3f}')
    opt_fpr = 1 - cl['specificity']; opt_tpr = cl['sensitivity']
    ax.scatter([opt_fpr], [opt_tpr], s=180, color='red', zorder=5,
               label=f"Optimal: thr={cl['threshold']:.3f}")
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{name} — порог по Юдену', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/07_roc_with_threshold.png', dpi=150, bbox_inches='tight')
plt.close()

# matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, cl, name in [(axes[0], cl_RF, 'Random Forest'),
                      (axes[1], cl_LR, 'Logistic L2')]:
    cm = np.array([[cl['tn'], cl['fp']], [cl['fn'], cl['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Control', 'Predicted Cancer'],
                yticklabels=['True Control', 'True Cancer'], ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_title(f"{name}\nSens={cl['sensitivity']:.2f}  Spec={cl['specificity']:.2f}  "
                 f"PPV={cl['ppv']:.2f}  NPV={cl['npv']:.2f}",
                 fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/08_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# Stability
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['Полный LOPO\n(3 проекта)', 'LOPO без\nPXD033741', 'Train: small\nTest: PXD033741']
rf_v = [obs_RF, auc_RF_s, auc_alt_RF]
lr_v = [obs_LR, auc_LR_s, auc_alt_LR]
xpos = np.arange(len(labels)); w = 0.38
ax.bar(xpos - w/2, rf_v, w, label='Random Forest', color='seagreen', alpha=0.85)
ax.bar(xpos + w/2, lr_v, w, label='Logistic L2', color='steelblue', alpha=0.85)
for i, (rv, lv) in enumerate(zip(rf_v, lr_v)):
    ax.text(i-w/2, rv+0.005, f'{rv:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i+w/2, lv+0.005, f'{lv:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel('AUROC'); ax.set_ylim(0.85, 1.02)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax.set_title('Стабильность: что если убрать доминирующий проект?', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/09_stability.png', dpi=150, bbox_inches='tight')
plt.close()

# Enrichment
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, e, title, col in [(axes[0], e1, 'Top-100 RF', 'Top100_overlap'),
                           (axes[1], e2, 'Universal markers', 'Universal_overlap')]:
    e_sort = e.sort_values('p_value').copy()
    e_sort['log10p'] = -np.log10(e_sort['p_value'].clip(lower=1e-300))
    colors = ['#E63946' if p < 0.05 else '#999999' for p in e_sort['p_value']]
    y_pos = np.arange(len(e_sort))
    ax.barh(y_pos, e_sort['log10p'].values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(e_sort['Hallmark'].values[::-1], fontsize=9)
    ax.set_xlabel('-log10(p-value)')
    ax.axvline(-np.log10(0.05), color='black', ls='--', alpha=0.6, label='p=0.05')
    ax.set_title(f'Hallmark enrichment: {title}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_FIGS}/10_enrichment.png', dpi=150, bbox_inches='tight')
plt.close()
