"""Task A: four-class and carcinoma-only subtype classification, five-fold CV and hold-out (Table 1, Tables S2, S2a, S2b).

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from EOC_DATA (default: ../../data); results go to EOC_OUT (default: ./results_revision).
"""
import os, numpy as np, pandas as pd, warnings
OUT=os.environ.get('EOC_OUT','results_revision')
def out(n): return os.path.join(OUT,n)
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
HAS_XGB=False   # the paper uses the scikit-learn implementation of gradient boosting in both tasks

RS=42; MISS=0.30; NPCA=50; KNN_K=5; NSPL=5
M=pd.read_csv(out('PXD033741_ibaq_269.csv.gz'),index_col=0)           # proteins x samples
sub=pd.read_csv(out('PXD033741_subtypes_269.csv'),index_col=0)['Subtype']
sub=sub.loc[M.columns]

def estimators(ntr):
    e={'LR_ElasticNet':LogisticRegression(penalty='elasticnet',l1_ratio=0.5,C=1.0,solver='saga',
          class_weight='balanced',max_iter=5000,random_state=RS),
       'SVM_RBF':SVC(kernel='rbf',C=1.0,class_weight='balanced',probability=True,random_state=RS),
       'Random Forest':RandomForestClassifier(n_estimators=500,class_weight='balanced',
          max_features='sqrt',random_state=RS,n_jobs=-1)}
    if HAS_XGB:
        e['XGBoost']=XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=4,subsample=0.9,
          eval_metric='mlogloss',random_state=RS,tree_method='hist',verbosity=0)
    else:
        e['Gradient Boosting']=GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,
          max_depth=3,random_state=RS)
    return e

def pipe(clf,ntr,nfeat):
    n=min(NPCA,ntr-1,nfeat)
    return Pipeline([('impute',KNNImputer(n_neighbors=KNN_K)),('scale',RobustScaler()),
                     ('pca',PCA(n_components=n,random_state=RS)),('clf',clf)])

def auroc_ovr(y,proba,classes):
    Yb=label_binarize(y,classes=classes); pres=Yb.sum(0)>0
    if pres.sum()<2: return np.nan
    return roc_auc_score(Yb[:,pres],proba[:,pres],average='macro',multi_class='ovr')

def run(label,classes):
    keep=sub.isin(classes).values
    Xr=M.values.T[keep]                      # samples x proteins
    y_raw=sub.values[keep]
    miss=np.isnan(Xr).mean(0); Xf=np.log2(Xr[:,miss<=MISS]+1.0)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder().fit(y_raw); y=le.transform(y_raw)
    Xtr,Xho,ytr,yho=train_test_split(Xf,y,test_size=0.20,stratify=y,random_state=RS)
    print(f'\n=== {label}: n={len(y)} ({dict(zip(*np.unique(y_raw,return_counts=True)))}), proteins {Xf.shape[1]}, train {len(ytr)}, hold-out {len(yho)}')
    skf=StratifiedKFold(n_splits=NSPL,shuffle=True,random_state=RS); rows=[]
    for name,clf in estimators(len(ytr)).items():
        ba,f1,au=[],[],[]
        for tr,te in skf.split(Xtr,ytr):
            p=pipe(clf,len(tr),Xf.shape[1]).fit(Xtr[tr],ytr[tr]); pr=p.predict(Xtr[te])
            ba.append(balanced_accuracy_score(ytr[te],pr)); f1.append(f1_score(ytr[te],pr,average='macro'))
            au.append(auroc_ovr(ytr[te],p.predict_proba(Xtr[te]),p.classes_))
        rows.append({'Analysis':label,'Model':name,'BalAcc':f'{np.mean(ba):.3f} ± {np.std(ba):.3f}',
            'F1_macro':f'{np.mean(f1):.3f} ± {np.std(f1):.3f}','AUROC_OvR':f'{np.nanmean(au):.3f} ± {np.nanstd(au):.3f}','_ba':np.mean(ba)})
        print(f'  {name:16s} BalAcc={np.mean(ba):.3f}±{np.std(ba):.3f} F1={np.mean(f1):.3f} AUROC={np.nanmean(au):.3f}')
    cv=pd.DataFrame(rows).sort_values('_ba',ascending=False)
    best=cv.iloc[0]['Model']
    bp=pipe(estimators(len(ytr))[best],len(ytr),Xf.shape[1]).fit(Xtr,ytr)
    pr=bp.predict(Xho); pa=bp.predict_proba(Xho)
    ho={'Analysis':label,'Model':best,'n_holdout':len(yho),'BalAcc':round(balanced_accuracy_score(yho,pr),3),
        'F1_macro':round(f1_score(yho,pr,average='macro'),3),'AUROC_OvR':round(auroc_ovr(yho,pa,bp.classes_),3)}
    print(f'  HOLD-OUT [{best}] n={len(yho)}: BalAcc={ho["BalAcc"]} F1={ho["F1_macro"]} AUROC={ho["AUROC_OvR"]}')
    labels=sorted(np.unique(np.concatenate([yho,pr]))); names=[le.classes_[i] for i in labels]
    cm=pd.DataFrame(confusion_matrix(yho,pr,labels=labels),index=names,columns=names)
    print(cm.to_string())
    # per-class accuracy in CV (pooled out-of-fold predictions)
    oof=np.empty(len(ytr),dtype=int)
    for tr,te in skf.split(Xtr,ytr):
        oof[te]=pipe(estimators(len(tr))[best],len(tr),Xf.shape[1]).fit(Xtr[tr],ytr[tr]).predict(Xtr[te])
    cmcv=pd.DataFrame(confusion_matrix(ytr,oof),index=le.classes_,columns=le.classes_)
    pc=pd.DataFrame({'Subtype':le.classes_,'n_train':cmcv.sum(1).values,
        'Correctly classified, CV (%)':(100*np.diag(cmcv)/cmcv.sum(1)).round(0).astype(int).values})
    print(pc.to_string(index=False))
    return cv.drop(columns='_ba'),pd.DataFrame([ho]),cm,cmcv,pc

cv4,ho4,cm4,cmcv4,pc4=run('Four classes (SC, CCC, EC, CT)',['SC','CCC','EC','CT'])
cv3,ho3,cm3,cmcv3,pc3=run('Carcinoma only (SC, CCC, EC)',['SC','CCC','EC'])
pd.concat([cv4,cv3]).to_csv(out('taskA_cv_metrics.csv'),index=False)
pd.concat([ho4,ho3]).to_csv(out('taskA_holdout_metrics.csv'),index=False)
cm4.to_csv(out('taskA_confusion_holdout_4class.csv')); cm3.to_csv(out('taskA_confusion_holdout_3class.csv'))
cmcv4.to_csv(out('taskA_confusion_cv_4class.csv')); cmcv3.to_csv(out('taskA_confusion_cv_3class.csv'))
pc4.to_csv(out('taskA_perclass_cv_4class.csv'),index=False); pc3.to_csv(out('taskA_perclass_cv_3class.csv'),index=False)
print('\nsaved')
