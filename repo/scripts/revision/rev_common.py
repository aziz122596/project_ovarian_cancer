"""Shared helpers: data loading, model definitions, parametric ComBat, CORAL.

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from the directory given by the environment variable
EOC_DATA (default: ../../data); results are written to EOC_OUT
(default: ./results_revision).
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

DATA_DIR=os.environ.get('EOC_DATA','../../data')
OUT=os.environ.get('EOC_OUT','results_revision'); os.makedirs(OUT,exist_ok=True)
# integrated matrix (sheets: matrix_after_project_z, sample_annotation); see data/README.md
DATA=os.path.join(DATA_DIR,os.environ.get('EOC_MATRIX','matrix_after_project_zscore_no_imputation.xlsx'))
if not os.path.exists(DATA):   # in this repository the matrix is stored in the repository root
    _alt=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..',os.path.basename(DATA))
    if os.path.exists(_alt): DATA=_alt
def out(name): return os.path.join(OUT,name)
RS=42; EV='PXD009655'
META=['Protein IDs','Gene names','Protein names','Majority protein IDs']

def load():
    df=pd.read_excel(DATA,sheet_name='matrix_after_project_z',engine='openpyxl')
    ann=pd.read_excel(DATA,sheet_name='sample_annotation',engine='openpyxl')
    cols=[c for c in df.columns if c not in META]
    full=df[cols].astype(float)
    core=(~full.isna()).all(axis=1)
    a=ann.set_index('SampleColumn').loc[cols]
    y=(a['Status']=='Cancer').astype(int).values
    g=a['Project'].values
    mat=np.where(g==EV,'EVs','Tissue')
    return df, full, core, cols, y, g, mat

def make_models():
    return {
        'LogReg L2': LogisticRegression(penalty='l2',C=1.0,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=RS),
        'LogReg L1': LogisticRegression(penalty='l1',C=1.0,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=RS),
        'Random Forest': RandomForestClassifier(n_estimators=500,class_weight='balanced',max_features='sqrt',random_state=RS,n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=RS),
        'SVM RBF': SVC(kernel='rbf',C=1.0,class_weight='balanced',probability=True,random_state=RS),
        'SVM Linear': SVC(kernel='linear',C=1.0,class_weight='balanced',probability=True,random_state=RS),
    }
def clone(b): return b.__class__(**b.get_params())
def proba(m,X): return m.predict_proba(X)[:,1]

# ---------- parametric ComBat (Johnson et al. 2007), reference-free ----------
def combat(X, batch, mod=None, eps=1e-8):
    """X: samples x features. batch: array of labels. mod: optional covariate matrix (samples x k).
    Returns adjusted X. Parametric empirical Bayes, standard algorithm."""
    X=np.asarray(X,float); n,p=X.shape
    batches=np.unique(batch); B=len(batches)
    D=np.zeros((n,B))
    for j,b in enumerate(batches): D[batch==b,j]=1
    design=D if mod is None else np.hstack([D,np.asarray(mod,float)])
    beta=np.linalg.lstsq(design,X,rcond=None)[0]           # (B+k) x p
    nb=D.sum(0)
    grand=(nb/n)@beta[:B]                                   # p
    fitted=design@beta
    var_pooled=((X-fitted)**2).sum(0)/n
    stand_mean=np.outer(np.ones(n),grand)
    if mod is not None:
        stand_mean=stand_mean+np.asarray(mod,float)@beta[B:]
    Z=(X-stand_mean)/np.sqrt(var_pooled+eps)
    gamma_hat=np.linalg.lstsq(D,Z,rcond=None)[0]           # B x p
    delta_hat=np.vstack([Z[batch==b].var(0,ddof=1) for b in batches])
    # EB priors
    def aprior(d): m=d.mean(); s2=d.var(); return (2*s2+m**2)/s2
    def bprior(d): m=d.mean(); s2=d.var(); return (m*s2+m**3)/s2
    gamma_star=np.zeros_like(gamma_hat); delta_star=np.zeros_like(delta_hat)
    for j,b in enumerate(batches):
        g_bar=gamma_hat[j].mean(); t2=gamma_hat[j].var()
        a_p=aprior(delta_hat[j]); b_p=bprior(delta_hat[j])
        nj=(batch==b).sum(); Zj=Z[batch==b]
        g_old=gamma_hat[j].copy(); d_old=delta_hat[j].copy()
        for _ in range(100):
            g_new=(t2*nj*gamma_hat[j]+d_old*g_bar)/(t2*nj+d_old)
            sum2=((Zj-g_new)**2).sum(0)
            d_new=(0.5*sum2+b_p)/(nj/2+a_p-1)
            if max(np.abs(g_new-g_old).max(),np.abs(d_new-d_old).max())<1e-5: break
            g_old,d_old=g_new,d_new
        gamma_star[j]=g_new; delta_star[j]=d_new
    out=Z.copy()
    for j,b in enumerate(batches):
        m=batch==b
        out[m]=(Z[m]-gamma_star[j])/np.sqrt(delta_star[j]+eps)
    return out*np.sqrt(var_pooled+eps)+stand_mean

# ---------- CORAL (Sun, Feng & Saenko 2016) ----------
def coral(Xs, Xt, lam=1.0):
    """Align source covariance to target covariance. Returns transformed source."""
    from scipy.linalg import sqrtm, inv
    Cs=np.cov(Xs,rowvar=False)+lam*np.eye(Xs.shape[1])
    Ct=np.cov(Xt,rowvar=False)+lam*np.eye(Xs.shape[1])
    A=np.real(inv(sqrtm(Cs))@sqrtm(Ct))
    return Xs@A
