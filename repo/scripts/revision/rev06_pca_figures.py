"""PCA of the core-protein analysis matrix (Figure 2) and exploratory PCA of the full
matrix with per-protein median imputation (Figure S1).

Part of the additional analyses added in revision (see README_revision.md).
"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rev_common import load, out

df,full,core,cols,y,g,mat=load()
Xc=full[core].T.values
Xf=full.T.values.copy(); med=np.nanmedian(Xf,axis=0); keep=~np.isnan(med); Xf=Xf[:,keep]; med=med[keep]
ii=np.where(np.isnan(Xf)); Xf[ii]=np.take(med,ii[1])
print('core matrix',Xc.shape,'| full matrix',Xf.shape,'(proteins absent from all samples dropped:',int((~keep).sum()),')')
pc={'PXD033741':'#d1495b','PXD012998':'#2e6f8e','PXD025864':'#7fb69c','PXD009655':'#edae49'}
def panel(X,name,title):
    pca=PCA(n_components=10,random_state=42).fit(X); P=pca.transform(X); ev=pca.explained_variance_ratio_*100
    print(f'{title}: PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%, top-10 {ev.sum():.1f}%')
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.52))
    for p,c in pc.items():
        m=g==p; ax[0].scatter(P[m,0],P[m,1],c=c,s=24,alpha=.9,edgecolor='white',linewidth=.3,label=p)
    ax[0].legend(title='PRIDE project',frameon=False,fontsize=8.5)
    for st,c,l in [(1,'#b5443b','Cancer'),(0,'#2e6f8e','Control')]:
        m=y==st; ax[1].scatter(P[m,0],P[m,1],c=c,s=24,alpha=.9,edgecolor='white',linewidth=.3,label=f'{l} (n={m.sum()})')
    m=mat=='EVs'; ax[1].scatter(P[m,0],P[m,1],facecolor='none',edgecolor='black',s=46,linewidth=.9,label='EVs (outlined)')
    ax[1].legend(title='Status',frameon=False,fontsize=8.5)
    for a,t in zip(ax,['(a)','(b)']):
        a.set_xlabel(f'PC1 ({ev[0]:.1f}%)'); a.set_ylabel(f'PC2 ({ev[1]:.1f}%)'); a.set_title(t,loc='left',fontweight='bold'); a.spines[['top','right']].set_visible(False)
    plt.tight_layout(); fig.savefig(out(name+'.png'),dpi=300); fig.savefig(out(name+'.tif'),dpi=900,pil_kwargs={'compression':'tiff_adobe_deflate'}); plt.close(fig)
panel(Xc,'Figure_2_PCA_core','Core matrix (Figure 2)')
panel(Xf,'Figure_S1_PCA_full','Full matrix, median imputation (Figure S1)')
