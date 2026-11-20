"""Permutation test for Task B (Table S10). Usage: python rev03_permutation_test.py 1000 [RF,LR_L2 | SVM_RBF,LR_L1]

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from the directory given by the environment variable
EOC_DATA (default: ../../data); results are written to EOC_OUT
(default: ./results_revision).
"""
import numpy as np, pandas as pd, time, sys
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from rev_common import *
N_PERM=int(sys.argv[1]); K=200
df,full,core,cols,y,g,mat=load(); Xc=full[core].T.values
tis=mat=='Tissue'; XT,yT,gT=Xc[tis],y[tis],g[tis]
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',max_features='sqrt',random_state=RS,n_jobs=-1)
def lr(): return LogisticRegression(penalty='l2',C=1.0,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=RS)
from sklearn.svm import SVC
def svm(): return SVC(kernel='rbf',C=1.0,class_weight='balanced',probability=True,random_state=RS)
def l1(): return LogisticRegression(penalty='l1',C=1.0,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=RS)
ALL={'RF':rf,'LR_L2':lr,'SVM_RBF':svm,'LR_L1':l1}
SEL=(sys.argv[2] if len(sys.argv)>2 else 'RF,LR_L2').split(','); MODELS=[(n,ALL[n]) for n in SEL]; TAG='_'.join(SEL)
def lopo_fs(X,y,gr,fn,k):
    yt,yp=[],[]
    for tr,te in LeaveOneGroupOut().split(X,y,gr):
        sel=SelectKBest(f_classif,k=k).fit(X[tr],y[tr]); m=fn().fit(sel.transform(X[tr]),y[tr])
        yp.extend(m.predict_proba(sel.transform(X[te]))[:,1]); yt.extend(y[te])
    return np.array(yt),np.array(yp)
obs={}
for n,fn in MODELS:
    a,b=lopo_fs(XT,yT,gT,fn,K); obs[n]=roc_auc_score(a,b)
print('observed',obs,flush=True)
rng=np.random.default_rng(42); null={n:[] for n,_ in MODELS}; t0=time.time()
for i in range(N_PERM):
    yp=rng.permutation(yT)
    for n,fn in MODELS:
        a,b=lopo_fs(XT,yp,gT,fn,K); null[n].append(roc_auc_score(a,b))
    if (i+1)%50==0: print(i+1,'perms',round(time.time()-t0),'s',{k:round(float(np.mean(v)),3) for k,v in null.items()},flush=True)
rows=[]
for n in null:
    v=np.array(null[n]); k=int((v>=obs[n]).sum()); rows.append({'Model':n,'AUROC_observed':obs[n],'n_perm':len(v),'k_ge_observed':k,'p_value':(k+1)/(len(v)+1),'null_mean':v.mean(),'null_sd':v.std(),'null_max':v.max(),'null_p95':np.percentile(v,95)})
pd.DataFrame(rows).to_csv(out(f'S10_permutation_{N_PERM}_{TAG}.csv'),index=False); pd.DataFrame(null).to_csv(out(f'S10_null_{N_PERM}_{TAG}.csv'),index=False)
print(pd.DataFrame(rows).to_string())
