# Графики результатов

В этой папке сохраняются все графики, генерируемые скриптами из `../scripts/`.

| Файл | Создаётся скриптом | Что показывает |
|---|---|---|
| `01_pca_visualization.png` | 01 | PCA по статусу и по проекту |
| `02_summary_heatmap.png` | 02 | AUROC heatmap 6 моделей × 4 подходов |
| `03_universal_markers_scatter.png` | 02 | Сонаправленность сигналов tissue vs EVs |
| `04_roc_best_per_approach.png` | 02 | ROC-кривые лучших моделей в каждом подходе |
| `05_bootstrap.png` | 03 | Bootstrap distribution AUROC |
| `06_permutation.png` | 03 | Permutation test (null vs observed) |
| `07_roc_with_threshold.png` | 03 | ROC с оптимальным порогом по Юдену |
| `08_confusion.png` | 03 | Confusion matrix |
| `09_stability.png` | 03 | Stability test без PXD033741 |
| `10_enrichment.png` | 03 | Hallmark enrichment |

Графики генерируются автоматически при запуске скриптов; в репозиторий не коммитятся (см. `.gitignore`).
