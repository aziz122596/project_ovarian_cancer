"""Benjamini-Hochberg correction of the tissue and vesicle comparisons and the two-stage
selection of concordant candidates (Section 2.4.2, Table 3, Table S6).

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from EOC_DATA (default: ../../data); results go to EOC_OUT.
"""
import numpy as np, pandas as pd
from scipy.stats import ttest_ind, spearmanr
from rev_common import load, out

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); q=p[o]*n/np.arange(1,n+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; r=np.empty(n); r[o]=np.minimum(q,1); return r

df,full,core,cols,y,g,mat=load()
X=full[core].T.values; tis=mat=='Tissue'; ev=~tis
def welch(Xa,Xb):
    """Welch t-test per protein. As in script 04, a protein is tested only if both groups
    contain more than one distinct value; otherwise it is not testable (t = 0, p = 1).
    A constant group arises when the protein was not detected in any sample of that group
    and all its intensities were set to the project minimum."""
    t=np.zeros(Xa.shape[1]); p=np.ones(Xa.shape[1]); ok=np.zeros(Xa.shape[1],bool)
    for i in range(Xa.shape[1]):
        if len(np.unique(Xa[:,i]))>1 and len(np.unique(Xb[:,i]))>1:
            t[i],p[i]=ttest_ind(Xa[:,i],Xb[:,i],equal_var=False); ok[i]=True
    return t,p,ok
tT,pT,okT=welch(X[tis&(y==1)],X[tis&(y==0)])
tE,pE,okE=welch(X[ev&(y==1)],X[ev&(y==0)])
res=pd.DataFrame({'Protein IDs':df.loc[core,'Protein IDs'].values,'Gene':df.loc[core,'Gene names'].astype(str).str.split(';').str[0].values,
                  't_tissue':tT,'p_tissue':pT,'q_tissue':bh(pT),'t_EV':tE,'p_EV':pE,'q_EV':bh(pE),'testable_in_EV':okE})
same=np.sign(res.t_tissue)==np.sign(res.t_EV)
res['nominal_both']=(res.p_tissue<0.05)&(res.p_EV<0.05)&(res.t_tissue.abs()>1.5)&(res.t_EV.abs()>1.5)&same   # criterion of the submitted version
res['concordant_candidate']=(res.q_tissue<0.05)&(res.p_EV<0.05)&(res.t_EV.abs()>1.5)&same
res.to_csv(out('S6_all_core_proteins_FDR.csv'),index=False)
res[res.concordant_candidate].sort_values('p_tissue').to_csv(out('S6_concordant_candidates.csv'),index=False)
print('tissue: p<0.05',int((res.p_tissue<0.05).sum()),'| q<0.05',int((res.q_tissue<0.05).sum()))
print('EV:     p<0.05',int((res.p_EV<0.05).sum()),'| q<0.05',int((res.q_EV<0.05).sum()),list(res.Gene[res.q_EV<0.05]))
print('concordant candidates:',int(res.concordant_candidate.sum()),'| up',int((res.concordant_candidate&(res.t_tissue>0)).sum()),'down',int((res.concordant_candidate&(res.t_tissue<0)).sum()))
print('not testable in EVs (constant group):',int((~okE).sum()),'| in tissue:',int((~okT).sum()))
print('nominal criterion of the submitted version:',int(res.nominal_both.sum()))
print('Spearman tissue vs EV t, testable proteins only: rho=%.3f p=%.2g n=%d'%(*spearmanr(res.t_tissue[okE],res.t_EV[okE]),int(okE.sum())))
print('Spearman, all 2701 with untestable set to t=0: rho=%.3f p=%.2g'%spearmanr(res.t_tissue,res.t_EV))
c=['SEC11A','TMCO1','SHMT2','ACAA1','MTA2','IL4I1','OCIAD2','DTX3L','THEM6']
print(res[res.Gene.isin(c)][['Gene','t_tissue','q_tissue','t_EV','p_EV','q_EV','concordant_candidate']].round(4).to_string(index=False))
