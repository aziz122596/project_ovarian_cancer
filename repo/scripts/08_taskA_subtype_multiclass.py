import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, accuracy_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_PATH      = '../data/PXD033741.xlsx'
OUTPUT_FIGS    = '../results/figures'
OUTPUT_TABLES  = '../results/tables'
RANDOM_STATE   = 42
CLASSES        = ['SC', 'CCC', 'EC', 'CT']
MISSING_THRESH = 0.30      
N_PCA          = 50
N_SPLITS       = 5
KNN_K          = 5
os.makedirs(OUTPUT_FIGS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

def load_real():
    df  = pd.read_excel(DATA_PATH, sheet_name='matrix', engine='openpyxl')
    ann = pd.read_excel(DATA_PATH, sheet_name='annotation', engine='openpyxl')
    meta = [c for c in df.columns
            if c in ('Protein IDs', 'Gene names', 'Protein names', 'Majority protein IDs')]
    sample_cols = [c for c in df.columns if c not in meta]
    ann = ann.set_index('SampleColumn').loc[sample_cols]
    X = df[sample_cols].astype(float).T.values           
    y = ann['Subtype'].astype(str).values
    return X, y


# def make_synthetic(n_per=(80, 64, 63, 30), n_prot=4300, seed=RANDOM_STATE):
#     rng = np.random.default_rng(seed)
#     ys, blocks = [], []
#     for cls, n in zip(CLASSES, n_per):
#         # у каждого класса свой сдвиг по случайному поднабору белков
#         base = rng.normal(0, 1, size=(n, n_prot))
#         sig_idx = rng.choice(n_prot, size=300, replace=False)
#         base[:, sig_idx] += rng.normal(2.0, 0.5, size=300)
#         blocks.append(base); ys += [cls] * n
#     X = np.vstack(blocks)
#     X = np.abs(X) * 1e5                                    # имитация интенсивностей
#     # вносим ~15 % пропусков
#     mask = rng.random(X.shape) < 0.15
#     X[mask] = np.nan
#     return X, np.array(ys)


if os.path.exists(DATA_PATH):
    print(f"[Реальные данные] {DATA_PATH}")
    X_raw, y_raw = load_real()
    SELFTEST = False
else:
    print("[SELF-TEST] data/PXD033741_subtypes.xlsx не найден — синтетика (числа НЕ научные).")
    X_raw, y_raw = make_synthetic()
    SELFTEST = True

print(f"  Матрица: {X_raw.shape[0]} образцов × {X_raw.shape[1]} белков")
for c in CLASSES:
    print(f"    {c}: {(y_raw == c).sum()}")

# 2. Фильтр белков по пропускам (без меток) + log2
miss_frac = np.isnan(X_raw).mean(axis=0)
keep = miss_frac <= MISSING_THRESH
X = np.log2(X_raw[:, keep] + 1.0)
print(f"  После фильтра >{int(MISSING_THRESH*100)}% пропусков: {X.shape[1]} белков")

# кодируем подтипы в целые 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)
NAME = {i: n for i, n in enumerate(le.classes_)}     
print(f"  Кодировка классов: {NAME}")

# 3. Train / hold-out 80/20 (стратифицированно)
X_tr, X_ho, y_tr, y_ho = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"  Train: {X_tr.shape[0]}   Hold-out: {X_ho.shape[0]}")

# 4. Модели 
def make_estimators():
    est = {
        'LR_ElasticNet': LogisticRegression(
            penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga',
            class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
        'SVM_RBF': SVC(kernel='rbf', C=1.0, class_weight='balanced',
                       probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            n_estimators=500, class_weight='balanced', max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        est['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, eval_metric='mlogloss', random_state=RANDOM_STATE,
            tree_method='hist', verbosity=0)
    else:
        est['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE)
    return est


def make_pipe(clf):
    n_comp = min(N_PCA, X_tr.shape[0] - 1, X_tr.shape[1])
    return Pipeline([
        ('impute', KNNImputer(n_neighbors=KNN_K)),
        ('scale',  RobustScaler()),
        ('pca',    PCA(n_components=n_comp, random_state=RANDOM_STATE)),
        ('clf',    clf),
    ])


def auroc_ovr(y_true, proba, classes):
    Yb = label_binarize(y_true, classes=classes)
    present = Yb.sum(axis=0) > 0
    if present.sum() < 2:
        return np.nan
    return roc_auc_score(Yb[:, present], proba[:, present],
                         average='macro', multi_class='ovr')

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, clf in make_estimators().items():
    ba, f1m, au = [], [], []
    for tr, te in skf.split(X_tr, y_tr):
        pipe = make_pipe(clf)
        pipe.fit(X_tr[tr], y_tr[tr])
        pred = pipe.predict(X_tr[te])
        ba.append(balanced_accuracy_score(y_tr[te], pred))
        f1m.append(f1_score(y_tr[te], pred, average='macro'))
        try:
            proba = pipe.predict_proba(X_tr[te])
            au.append(auroc_ovr(y_tr[te], proba, pipe.classes_))
        except Exception:
            au.append(np.nan)
    cv_rows.append({
        'Model': name,
        'BalAcc':   f"{np.mean(ba):.3f} ± {np.std(ba):.3f}",
        'F1_macro': f"{np.mean(f1m):.3f} ± {np.std(f1m):.3f}",
        'AUROC_OvR':f"{np.nanmean(au):.3f} ± {np.nanstd(au):.3f}",
        '_ba': np.mean(ba),
    })
    print(f"  {name:16s} BalAcc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
          f"F1={np.mean(f1m):.3f}  AUROC={np.nanmean(au):.3f}")

cv_df = pd.DataFrame(cv_rows).sort_values('_ba', ascending=False)
best_name = cv_df.iloc[0]['Model']
cv_df.drop(columns='_ba').to_csv(f'{OUTPUT_TABLES}/taskA_cv_metrics.csv', index=False)

best_pipe = make_pipe(make_estimators()[best_name]).fit(X_tr, y_tr)
pred_ho = best_pipe.predict(X_ho)
proba_ho = best_pipe.predict_proba(X_ho)

ho = {
    'Model': best_name,
    'BalAcc':   round(balanced_accuracy_score(y_ho, pred_ho), 3),
    'F1_macro': round(f1_score(y_ho, pred_ho, average='macro'), 3),
    'AUROC_OvR':round(auroc_ovr(y_ho, proba_ho, best_pipe.classes_), 3),
}
pd.DataFrame([ho]).to_csv(f'{OUTPUT_TABLES}/taskA_holdout_metrics.csv', index=False)
print(f"\n  HOLD-OUT [{best_name}]: BalAcc={ho['BalAcc']}  "
      f"F1={ho['F1_macro']}  AUROC={ho['AUROC_OvR']}")

labels = [i for i in sorted(np.unique(y_ho))]
names  = [NAME[i] for i in labels]
cm = confusion_matrix(y_ho, pred_ho, labels=labels)
per_class = []
for i, c in enumerate(names):
    denom = cm[i].sum()
    per_class.append({'Class': c, 'Accuracy': round(cm[i, i] / denom, 3) if denom else np.nan})
pd.DataFrame(per_class).to_csv(f'{OUTPUT_TABLES}/taskA_per_class_accuracy.csv', index=False)

fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Task A confusion — {best_name}' + (' [SELF-TEST]' if SELFTEST else ''))
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f'{OUTPUT_FIGS}/taskA_confusion_{best_name}.png', dpi=200)
plt.close(fig)
