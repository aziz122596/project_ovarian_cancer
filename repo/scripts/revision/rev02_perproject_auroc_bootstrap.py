"""Per-project AUROC in leave-one-project-out validation with bootstrap 95% CI,
unweighted (macro) mean and pooled AUROC for the six models (Table 2, Tables S8b, S8c).

Part of the additional analyses added in revision (see README_revision.md).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from rev_common import *

df,full,core,cols,y,g,mat=load(); Xc=full[core].T.values; tis=mat=='Tissue'
XT,yT,gT=Xc[tis],y[tis],g[tis]; rng=np.random.default_rng(42); rows=[]; summ=[]
for n,b in make_models().items():
    yt_all,yp_all=[],[]; per={}
    for tr,te in LeaveOneGroupOut().split(XT,yT,gT):
        m=clone(b).fit(XT[tr],yT[tr]); p=proba(m,XT[te]); yt=yT[te]; a=roc_auc_score(yt,p); bs=[]
        for _ in range(2000):
            idx=rng.integers(0,len(yt),len(yt))
            if len(set(yt[idx]))<2: continue
            bs.append(roc_auc_score(yt[idx],p[idx]))
        rows.append({'Model':n,'Held_out':gT[te][0],'n':len(yt),'n_cancer':int(yt.sum()),'n_control':int((1-yt).sum()),
                     'AUROC':a,'CI_low':np.percentile(bs,2.5),'CI_high':np.percentile(bs,97.5)})
        per[gT[te][0]]=a; yt_all+=list(yt); yp_all+=list(p)
    summ.append({'Model':n,**per,'Macro_mean':np.mean(list(per.values())),'Pooled':roc_auc_score(yt_all,yp_all)})
pd.DataFrame(rows).to_csv(out('S8c_perproject_bootstrapCI.csv'),index=False)
pd.DataFrame(summ).to_csv(out('S8b_auroc_per_project_vs_pooled.csv'),index=False)
print(pd.DataFrame(summ).round(3).to_string(index=False))
