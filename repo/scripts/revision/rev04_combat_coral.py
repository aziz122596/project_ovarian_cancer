"""Batch-correction and domain-adaptation benchmark: ComBat (two variants) and CORAL in LOPO and in tissue-to-vesicle transfer (Table S12).

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from the directory given by the environment variable
EOC_DATA (default: ../../data); results are written to EOC_OUT
(default: ./results_revision).
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from rev_common import *
df,full,core,cols,y,g,mat=load(); Xc=full[core].T.values
tis=mat=='Tissue'; XT,yT,gT=Xc[tis],y[tis],g[tis]; XE,yE=Xc[~tis],y[~tis]
# A. ComBat without covariate on z-scored matrix
Xa=combat(XT,gT); print('A. ComBat(no mod) on per-project z-scored tissue matrix: max|diff| =',np.abs(Xa-XT).max().round(4),' mean|diff| =',np.abs(Xa-XT).mean().round(5))
models={k:v for k,v in make_models().items() if k in ('LogReg L2','Random Forest','SVM Linear')}
rows=[]
def run(label,prep):
    for n,b in models.items():
        yt,yp,per={}, {}, {}
        yt,yp=[],[]
        for tr,te in LeaveOneGroupOut().split(XT,yT,gT):
            Xtr,Xte=prep(XT[tr],yT[tr],gT[tr],XT[te])
            m=clone(b).fit(Xtr,yT[tr]); p=proba(m,Xte); yt.extend(yT[te]); yp.extend(p)
            per[gT[te][0]]=roc_auc_score(yT[te],p)
        pooled=roc_auc_score(np.array(yt),np.array(yp))
        # EV transfer
        Xtr,Xte=prep(XT,yT,gT,XE); m=clone(b).fit(Xtr,yT); ev=roc_auc_score(yE,proba(m,Xte))
        r={'Setting':label,'Model':n,'LOPO_pooled_AUROC':pooled,**{f'AUROC_{k}':v for k,v in per.items()},'EV_transfer_AUROC':ev}; rows.append(r); print(label,n,round(pooled,3),{k:round(v,3) for k,v in per.items()},'EV',round(ev,3),flush=True)
def p_base(Xtr,ytr,gtr,Xte): return Xtr,Xte
def p_combat_mod(Xtr,ytr,gtr,Xte):
    # ComBat with status covariate on training projects only; held-out project untouched (its own z-scoring)
    return combat(Xtr,gtr,mod=ytr.reshape(-1,1)),Xte
def p_coral(Xtr,ytr,gtr,Xte):
    # unsupervised: align training covariance to held-out project covariance (labels of test never used)
    return coral(Xtr,Xte,lam=1.0),Xte
def p_combat_transductive(Xtr,ytr,gtr,Xte):
    # ComBat over train+test batches without covariate (labels unused) -> identity on z-scored data; included for completeness
    Xall=np.vstack([Xtr,Xte]); ball=np.concatenate([gtr,np.array(['HELDOUT']*len(Xte))])
    Xadj=combat(Xall,ball); return Xadj[:len(Xtr)],Xadj[len(Xtr):]
t0=time.time()
run('Baseline (per-project z-score)',p_base)
run('ComBat transductive, no covariate',p_combat_transductive)
run('ComBat on training projects, status covariate',p_combat_mod)
run('CORAL alignment to held-out project',p_coral)
pd.DataFrame(rows).to_csv(out('S12_domain_adaptation.csv'),index=False); print('done',round(time.time()-t0),'s')
