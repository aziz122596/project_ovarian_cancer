"""
Скрипт 4: Выгрузка списков белков с UniProt IDs
================================================

ВХОД:  data/matrix_after_project_zscore_no_imputation.xlsx
ВЫХОД: results/protein_lists/all_core_proteins_with_metrics.csv
       results/protein_lists/top100_RF_proteins.csv
       results/protein_lists/universal_markers_tissue_EVs.csv
       results/protein_lists/PROTEIN_LIST_FULL.xlsx

Что делает:
  1. Извлекает UniProt IDs, Gene names, Protein names из исходного файла
  2. Считает RF importance, LogReg L1/L2 коэффициенты, t-stats для tissue и EVs
  3. Размечает universal markers (значимые в обоих матриксах)
  4. Аннотирует hallmark-категории
  5. Сохраняет в 3 CSV + единый Excel со всеми листами
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

DATA_PATH = '../data/matrix_after_project_zscore_no_imputation.xlsx'
OUTPUT_DIR = '../results/protein_lists'
RANDOM_STATE = 42
EV_PROJECT = 'PXD009655'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name='matrix_after_project_z', engine='openpyxl')
ann = pd.read_excel(DATA_PATH, sheet_name='sample_annotation', engine='openpyxl')
META = ['Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs']
sample_cols = [c for c in df.columns if c not in META]
ann['Material'] = ann['Project'].apply(lambda p: 'EVs' if p == EV_PROJECT else 'Tissue')

data = df[sample_cols].astype(float)
core_mask = (~data.isna()).all(axis=1)
df_core = df[core_mask].reset_index(drop=True)

def first_id(s):
    """Первый ID из MaxQuant-формата 'P12345;P67890;...'"""
    if pd.isna(s):
        return ''
    return str(s).strip().split(';')[0]

def all_ids(s):
    if pd.isna(s):
        return ''
    return str(s).strip()

df_core['UniProt_ID'] = df_core['Protein IDs'].apply(first_id)
df_core['UniProt_All_IDs'] = df_core['Protein IDs'].apply(all_ids)
df_core['Gene_first'] = df_core['Gene names'].apply(first_id)
df_core['Gene_all'] = df_core['Gene names'].apply(all_ids)
df_core['Protein_name'] = df_core['Protein names'].fillna('')
df_core['Majority_protein_IDs'] = df_core['Majority protein IDs'].fillna('').apply(all_ids)
print(f"Коровых белков: {len(df_core)}")

feat_names = [
    str(r['Gene_first']) if r['Gene_first'] and r['Gene_first'] != 'nan'
    else str(r['UniProt_ID'])
    for _, r in df_core.iterrows()
]

X_full = df_core[sample_cols].astype(float).T.values
ann_idx = ann.set_index('SampleColumn').loc[sample_cols]
y_full = (ann_idx['Status'] == 'Cancer').astype(int).values
material_full = ann_idx['Material'].values

tissue_mask = material_full == 'Tissue'
ev_mask = material_full == 'EVs'
X_T = X_full[tissue_mask]; y_T = y_full[tissue_mask]
X_E = X_full[ev_mask]; y_E = y_full[ev_mask]

print("Fit Random Forest...")
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                             max_features='sqrt',
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_T, y_T)
rf_importance = rf.feature_importances_

print("Fit Logistic L1 / L2...")
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0,
                            class_weight='balanced', max_iter=2000,
                            random_state=RANDOM_STATE)
lr_l1.fit(X_T, y_T)
lr_l1_coef = lr_l1.coef_[0]

lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0,
                            class_weight='balanced', max_iter=2000,
                            random_state=RANDOM_STATE)
lr_l2.fit(X_T, y_T)
lr_l2_coef = lr_l2.coef_[0]

print("Compute t-stats...")
def t_stats(X, y):
    ts = np.zeros(X.shape[1]); ps = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        a, b = X[y == 1, i], X[y == 0, i]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            ts[i], ps[i] = t, p
    return ts, ps

t_T, p_T = t_stats(X_T, y_T)
t_E, p_E = t_stats(X_E, y_E)

universal_mask = (
    (p_T < 0.05) & (p_E < 0.05) &
    (np.sign(t_T) == np.sign(t_E)) &
    (np.abs(t_T) > 1.5) & (np.abs(t_E) > 1.5)
)

# Hallmark-категории

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

def annotate_hallmark(gene):
    cats = [hn for hn, genes in hallmark_sets.items() if gene in genes]
    return '; '.join(cats)


master = pd.DataFrame({
    'Gene_name': df_core['Gene_first'],
    'Gene_all_synonyms': df_core['Gene_all'],
    'UniProt_ID': df_core['UniProt_ID'],
    'UniProt_All_IDs': df_core['UniProt_All_IDs'],
    'Majority_protein_IDs': df_core['Majority_protein_IDs'],
    'Protein_name': df_core['Protein_name'],
    'RF_importance': rf_importance,
    'LogReg_L1_coef': lr_l1_coef,
    'LogReg_L2_coef': lr_l2_coef,
    't_stat_Tissue': t_T,
    'p_value_Tissue': p_T,
    't_stat_EVs': t_E,
    'p_value_EVs': p_E,
    'is_universal_marker': universal_mask,
    'Direction_in_Cancer': ['↑ Up' if t > 0 else '↓ Down' if t < 0 else '—' for t in t_T],
})
master['Hallmark_category'] = master['Gene_name'].apply(annotate_hallmark)
master = master.sort_values('RF_importance', ascending=False).reset_index(drop=True)
master.insert(0, 'Rank_RF', range(1, len(master) + 1))

top100 = master.head(100).copy()
universal_df = master[master['is_universal_marker']].sort_values('p_value_Tissue').reset_index(drop=True)

master.to_csv(f'{OUTPUT_DIR}/all_core_proteins_with_metrics.csv',
              index=False, encoding='utf-8-sig')
top100.to_csv(f'{OUTPUT_DIR}/top100_RF_proteins.csv',
              index=False, encoding='utf-8-sig')
universal_df.to_csv(f'{OUTPUT_DIR}/universal_markers_tissue_EVs.csv',
                     index=False, encoding='utf-8-sig')

with pd.ExcelWriter(f'{OUTPUT_DIR}/PROTEIN_LIST_FULL.xlsx', engine='openpyxl') as writer:
    top100.to_excel(writer, sheet_name='Top100_RF_importance', index=False)
    universal_df.to_excel(writer, sheet_name='Universal_markers', index=False)
    master.to_excel(writer, sheet_name='All_core_proteins', index=False)
    in_hallmark = master[master['Hallmark_category'] != '']
    in_hallmark.to_excel(writer, sheet_name='Hallmark_annotated', index=False)
