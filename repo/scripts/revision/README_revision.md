# Additional analyses added in revision

Scripts in this folder reproduce the analyses added to the manuscript in revision.
They do not replace scripts `01`–`09` of the primary analysis.

## Environment

Two environments are documented with pinned versions:

| File | Used for | Versions |
|---|---|---|
| `../../requirements.txt` | primary analysis (scripts 01–09) | Python 3.13.16, scikit-learn 1.6.1, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3 |
| `../../requirements-revision.txt` | scripts in this folder | Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 |

`random_state = 42` is used throughout.

## Input data

Scripts read input from the directory given by the environment variable `EOC_DATA`
(default `../../data`) and write results to `EOC_OUT` (default `./results_revision`).

| File expected in `EOC_DATA` | Content | Needed by |
|---|---|---|
| `matrix_after_project_zscore_no_imputation.xlsx` | integrated protein matrix after per-project z-scoring (sheets `matrix_after_project_z`, `sample_annotation`) | rev01–rev07 |
| `proteinGroups_PXD033741.csv` | MaxQuant `proteinGroups` table of PXD033741, exported as CSV | rev07 |
| `Gong2023_SupplementaryData1.xlsx` | Supplementary Data 1 of Gong et al., *Nat. Commun.* 2023 (sample → histological type) | rev07 |

The name of the matrix file can be changed with `EOC_MATRIX`. If the matrix is not found in `EOC_DATA`, the scripts look for it in the repository root, where it is stored in this repository. The two other files are not redistributed here: the `proteinGroups` table is obtained by reprocessing the RAW files of PXD033741, and Supplementary Data 1 is available from the publisher's page of Gong et al.

## Scripts

| Script | What it does | Manuscript item |
|---|---|---|
| `rev_common.py` | data loading, model definitions, parametric ComBat, CORAL | – |
| `rev01_fdr_candidates.py` | Welch t-tests, Benjamini–Hochberg correction, two-stage selection of concordant candidates | Section 2.4.2, Table 3, Table S6 |
| `rev02_perproject_auroc_bootstrap.py` | per-project AUROC in LOPO with bootstrap 95% CI, macro mean, pooled AUROC | Table 2, Tables S8b, S8c |
| `rev03_permutation_test.py` | permutation test of the full LOPO scheme | Table S10 |
| `rev04_combat_coral.py` | ComBat (two variants) and CORAL in LOPO and in tissue-to-vesicle transfer | Table S12 |
| `rev05_minfill_sensitivity.py` | sensitivity to the minimum-value substitution of non-detected proteins | Section 2.5, Table S13 |
| `rev06_pca_figures.py` | PCA of the core matrix and exploratory PCA of the full matrix | Figure 2, Figure S1 |
| `rev07_taskA_build_matrix.py` | PXD033741 iBAQ matrix (269 samples) with subtype labels | Section 2.2 |
| `rev08_taskA_classification.py` | four-class and carcinoma-only subtype classification, CV and hold-out | Table 1, Tables S2, S2a, S2b |
| `rev09_figure3_subtypes.py` | PCA and t-SNE of the Task A training set | Figure 3 |

## How to run

```bash
pip install -r ../../requirements-revision.txt
export EOC_DATA=../../data
python rev01_fdr_candidates.py
python rev02_perproject_auroc_bootstrap.py
python rev03_permutation_test.py 1000 RF,LR_L2
python rev03_permutation_test.py 1000 SVM_RBF,LR_L1
python rev04_combat_coral.py
python rev05_minfill_sensitivity.py
python rev06_pca_figures.py
python rev07_taskA_build_matrix.py      # must be run before rev08 and rev09
python rev08_taskA_classification.py
python rev09_figure3_subtypes.py
```

Approximate run time on a laptop: rev03 about 10 minutes per call, rev08 about 10 minutes,
the others from seconds to a few minutes.

## Notes

- **Testability rule (rev01).** As in script `04`, a protein is tested only if both groups contain
  more than one distinct value. A constant group arises when the protein was not detected in any
  sample of that group and all its intensities were set to the project minimum. No protein is
  affected in tissue; 254 proteins are affected in the vesicle dataset. They are treated as not
  testable (p = 1) and stay in the denominator of the Benjamini–Hochberg correction.
- **Gradient boosting.** Both tasks use the scikit-learn implementation (`GradientBoostingClassifier`).
- **Expected output of rev01:** tissue 2245 proteins at p < 0.05 and 2216 at q < 0.05; vesicles 247 and 3;
  193 concordant candidates (186 up, 7 down).
- **Random forest.** Pooled AUROC of the random forest differs in the third decimal between library
  versions (0.959 in the primary analysis, 0.953 with the revision environment); the other models are
  identical.
- The over-representation analysis (Table S11) was run in the g:Profiler web tool (g:GOSt; GO BP, KEGG,
  Reactome; g:SCS < 0.05) with the 193 concordant candidates as query and the 2,701 core proteins as
  custom background.
