"""Sensitivity of the LOPO classification and of the nine candidates to the minimum-value substitution of non-detected proteins (Table S13).

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from the directory given by the environment variable
EOC_DATA (default: ../../data); results are written to EOC_OUT
(default: ./results_revision).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from scipy.stats import ttest_ind
from rev_common import *
df,full,core,cols,y,g,mat=load(); Xc=full[core].T.values  # samples x proteins
tis=mat=='Tissue'; XT,yT,gT=Xc[tis],y[tis],g[tis]; XE,yE=Xc[~tis],y[~tis]
# detect min-filled cells: within project & protein, value == project minimum AND that minimum occurs >1 times
flag=np.zeros_like(Xc,dtype=bool)
for p in np.unique(g):
    m=g==p; sub=Xc[m]; mn=sub.min(0); ties=(np.isclose(sub,mn)).sum(0)
    flag[m]=np.isclose(sub,mn)&(ties>1)
print('min-filled cells overall: %.1f%%'%(100*flag.mean()))
for p in np.unique(g): print(' ',p,'%.1f%%'%(100*flag[g==p].mean()))
prot_any=flag.any(0); print('proteins with >=1 min-filled cell:',prot_any.sum(),'of',Xc.shape[1],'; strict core:',(~prot_any).sum())
genes=df.loc[core,'Gene names'].astype(str).str.split(';').str[0].values
cands=['SEC11A','TMCO1','SHMT2','ACAA1','MTA2','IL4I1','OCIAD2','DTX3L','THEM6']
ci=[np.where(genes==c)[0][0] for c in cands]
print('min-filled fraction in 9 candidates (tissue / EV):')
for c,i in zip(cands,ci): print('  %-7s tissue %.1f%%  EV %.1f%%'%(c,100*flag[tis,i].mean(),100*flag[~tis,i].mean()))
# variants
def lopo(X,models):
    out={}
    for n,b in models.items():
        yt,yp=[],[]
        for tr,te in LeaveOneGroupOut().split(X,yT,gT):
            m=clone(b).fit(X[tr],yT[tr]); yp.extend(proba(m,X[te])); yt.extend(yT[te])
        out[n]=roc_auc_score(yt,yp)
    return out
models={k:v for k,v in make_models().items() if k in ('LogReg L2','Random Forest')}
rows=[]
# A baseline
r=lopo(XT,models); rows.append({'Variant':'Baseline core (2701, min-fill as is)','n_proteins':XT.shape[1],**r}); print(rows[-1])
# B strict core
Xs=XT[:,~prot_any]; r=lopo(Xs,models); rows.append({'Variant':'Strict core (no min-filled cell in any sample)','n_proteins':Xs.shape[1],**r}); print(rows[-1])
# C KNN imputation of min-filled cells (per project, fitted within training folds only -> here: per project transductive, labels unused)
Xk=XT.copy().astype(float); Xk[flag[tis]]=np.nan
Xk2=Xk.copy()
for p in np.unique(gT):
    m=gT==p; Xk2[m]=KNNImputer(n_neighbors=5).fit_transform(Xk[m])
r=lopo(Xk2,models); rows.append({'Variant':'KNN (k=5) imputation of min-filled cells, per project','n_proteins':XT.shape[1],**r}); print(rows[-1])
# D drop proteins with >10% min-filled in any tissue project
bad=np.zeros(Xc.shape[1],bool)
for p in np.unique(gT): bad|=flag[g==p].mean(0)>0.10
Xd=XT[:,~bad]; r=lopo(Xd,models); rows.append({'Variant':'Drop proteins with >10% min-filled cells in any tissue project','n_proteins':Xd.shape[1],**r}); print(rows[-1])
pd.DataFrame(rows).to_csv(out('S13_minfill_sensitivity.csv'),index=False)
# candidates: t-stats with min-filled cells set to NaN (dropped) vs as-is
print('candidate t-stats tissue: as-is vs min-filled removed')
for c,i in zip(cands,ci):
    a=XT[:,i]; t1=ttest_ind(a[yT==1],a[yT==0],equal_var=False).statistic
    b=a.copy(); fl=flag[tis,i]; t2=ttest_ind(b[(yT==1)&~fl],b[(yT==0)&~fl],equal_var=False).statistic
    print('  %-7s %6.2f -> %6.2f  (dropped %d cells)'%(c,t1,t2,fl.sum()))
