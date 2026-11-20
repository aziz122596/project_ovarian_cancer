"""Figure 3: PCA and t-SNE of the Task A training set.

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from EOC_DATA (default: ../../data); results go to EOC_OUT (default: ./results_revision).
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer; from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA; from sklearn.manifold import TSNE; from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
import os
OUT=os.environ.get('EOC_OUT','results_revision')
M=pd.read_csv(os.path.join(OUT,'PXD033741_ibaq_269.csv.gz'),index_col=0); sub=pd.read_csv(os.path.join(OUT,'PXD033741_subtypes_269.csv'),index_col=0)['Subtype'].loc[M.columns]
X=M.values.T; miss=np.isnan(X).mean(0); X=np.log2(X[:,miss<=0.30]+1.0); y=LabelEncoder().fit_transform(sub.values); lab=sub.values
Xtr,Xho,ltr,lho=train_test_split(X,lab,test_size=0.20,stratify=y,random_state=42)
Z=RobustScaler().fit_transform(KNNImputer(n_neighbors=5).fit_transform(Xtr))
p=PCA(n_components=50,random_state=42).fit(Z); P=p.transform(Z); ev=p.explained_variance_ratio_*100
T=TSNE(n_components=2,perplexity=30,random_state=42,init='pca').fit_transform(P)
col={'SC':('#e05263','Serous Carcinoma'),'CCC':('#4f7cac','Clear Cell Carcinoma'),'EC':('#3fa796','Endometrioid Carcinoma'),'CT':('#f4a259','Normal Control')}
fig,ax=plt.subplots(1,2,figsize=(11,4.65))
for k,(c,n) in col.items():
    m=ltr==k
    ax[0].scatter(P[m,0],P[m,1],c=c,s=26,alpha=.9,edgecolor='white',linewidth=.3,label=f'{n} (n={m.sum()})')
    ax[1].scatter(T[m,0],T[m,1],c=c,s=26,alpha=.9,edgecolor='white',linewidth=.3,label=f'{n} (n={m.sum()})')
ax[0].set_xlabel(f'PC1 ({ev[0]:.1f}%)'); ax[0].set_ylabel(f'PC2 ({ev[1]:.1f}%)'); ax[1].set_xlabel('t-SNE 1'); ax[1].set_ylabel('t-SNE 2')
for a,t in zip(ax,['(a)','(b)']): a.set_title(t,loc='left',fontweight='bold'); a.legend(fontsize=7.5,frameon=True); a.grid(alpha=.2)
plt.tight_layout(); fig.savefig(os.path.join(OUT,'Figure_3.png'),dpi=300,facecolor='white'); fig.savefig(os.path.join(OUT,'Figure_3.tif'),dpi=900,facecolor='white',pil_kwargs={'compression':'tiff_adobe_deflate'})
print('n',len(ltr),dict(zip(*np.unique(ltr,return_counts=True))),'ev',ev[:2].round(1))
