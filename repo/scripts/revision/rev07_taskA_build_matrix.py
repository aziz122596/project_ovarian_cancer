"""Task A: build the PXD033741 iBAQ matrix (269 samples) and attach histological subtype labels.

Part of the additional analyses added in revision (see README_revision.md).
Input files are read from EOC_DATA (default: ../../data); results go to EOC_OUT (default: ./results_revision).
"""
import csv, os, numpy as np, pandas as pd, openpyxl
D=os.environ.get('EOC_DATA','../../data'); OUT=os.environ.get('EOC_OUT','results_revision'); os.makedirs(OUT,exist_ok=True)
_mx=os.path.join(D,os.environ.get('EOC_MATRIX','matrix_after_project_zscore_no_imputation.xlsx'))
if not os.path.exists(_mx): _mx=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..',os.path.basename(_mx))   # repository root
# inputs (see data/README.md):
#  proteinGroups_PXD033741.csv  - MaxQuant proteinGroups table of PXD033741 exported as CSV
#  Gong2023_SupplementaryData1.xlsx - Supplementary Data 1 of Gong et al., Nat. Commun. 2023 (sample -> histological type)
#  matrix_after_project_zscore_no_imputation.xlsx - integrated matrix (sheet sample_annotation: control samples)
f=os.path.join(D,'proteinGroups_PXD033741.csv')
wb=openpyxl.load_workbook(os.path.join(D,'Gong2023_SupplementaryData1.xlsx'),read_only=True); ws=wb.worksheets[0]
ann={r[0]:r[1] for i,r in enumerate(ws.iter_rows(values_only=True)) if i>=2 and r[0] and r[1]}
rows=[]; 
with open(f,newline='',encoding='utf-8',errors='ignore') as fh:
    rd=csv.reader(fh); hdr=next(rd)
    idx={h:i for i,h in enumerate(hdr)}
    ib=[(h[5:],i) for h,i in idx.items() if h.startswith('iBAQ ') and h!='iBAQ peptides' and h!='iBAQ']
    rev,con,site=idx['Reverse'],idx['Potential contaminant'],idx['Only identified by site']
    pid,gn,pn=idx['Protein IDs'],idx['Gene names'],idx['Protein names']
    bad=0
    for r in rd:
        if len(r)<len(hdr): continue
        if r[rev]=='+' or r[con]=='+' or r[site]=='+': continue
        vals=[]
        for s,i in ib:
            v=r[i].strip().replace(',','.')
            try: x=float(v)
            except: x=np.nan; bad+=1
            vals.append(x)
        rows.append([r[pid],r[gn],r[pn]]+vals)
print('bad cells',bad)
samples=[s for s,_ in ib]
df=pd.DataFrame(rows,columns=['Protein IDs','Gene names','Protein names']+samples)
print('proteins after filter',len(df),'samples',len(samples))
# subtype annotation: 239 tumours from SD1, 30 controls = the analysis matrix controls, drop F1712329
m=openpyxl.load_workbook(_mx,read_only=True)['sample_annotation']
mr=list(m.iter_rows(values_only=True)); ctrl=[r[0] for r in mr[1:] if r[1]=='PXD033741' and r[2]!='Cancer']
sub={}
for s in samples:
    if s in ann: sub[s]=ann[s]
    elif s in ctrl: sub[s]='CT'
print('annotated',len(sub),pd.Series(sub).value_counts().to_dict(),'excluded',[s for s in samples if s not in sub])
keep=[s for s in samples if s in sub]
X=df[keep].values.astype(float); X[X==0]=np.nan
out=pd.DataFrame(X,index=df['Protein IDs'],columns=keep)
out.to_csv(os.path.join(OUT,'PXD033741_ibaq_269.csv.gz'),compression='gzip'); pd.Series(sub).loc[keep].to_csv(os.path.join(OUT,'PXD033741_subtypes_269.csv'),header=['Subtype'])
df[['Protein IDs','Gene names','Protein names']].to_csv(os.path.join(OUT,'PXD033741_protein_meta.csv'),index=False)
print('matrix',out.shape,'missing frac',np.isnan(X).mean().round(3))
