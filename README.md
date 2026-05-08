# Ovarian Cancer Classification

Классификация рака яичников (Cancer vs Control) на основе протеомных данных из 4 независимых проектов PRIDE с использованием отбора коровых белков (coverage 100%).

## Краткая сводка результатов

- **180 образцов** из 4 проектов PRIDE (130 ткань + 50 микровезикул)
- **2 701 коровых белков** отобрано из 9 610 (coverage 100%)
- **Random Forest на ткани**: AUROC = 0.957 [95% CI: 0.92–0.98], permutation p = 0.010
- 4 стратегии интеграции данных × 6 алгоритмов ML = 24 модели сравнены
- Топ-белки значимо обогащены онкогенными путями ER stress (FE = 10.3×), OXPHOS (FE = 8.2×), apoptosis (FE = 13.5×)

## Структура репозитория

```
.
├── data/                      # входные данные (см. data/README.md)
├── scripts/                   # скрипты по этапам анализа
│   ├── 01_core_proteins_selection.py
│   ├── 02_four_approaches_comparison.py
│   ├── 03_validation_extended.py
│   ├── 04_protein_lists_export.py
│   ├── 05_direction_analysis.py
│   ├── 06_internal_validation_ids.py
│   └── 07_external_validation_prep.py
├── notebooks/
│   └── full_pipeline_colab.py # объединённый ноутбук для Google Colab
├── results/
│   ├── figures/               # графики (PNG)
│   ├── tables/                # сводные таблицы (CSV)
│   └── protein_lists/         # списки белков с UniProt IDs
└── docs/
    ├── METHODOLOGY.md         # методология
    └── RESULTS.md             # подробные результаты
```

## Требования

```bash
pip install -r requirements.txt
```

Python 3.9+, основные пакеты: scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, openpyxl.

## Воспроизводимость

1. Скачать входной файл
2. Запустите скрипты из `scripts/` по порядку:

```bash
cd scripts/
python 01_core_proteins_selection.py
python 02_four_approaches_comparison.py
python 03_validation_extended.py
python 04_protein_lists_export.py
python 05_direction_analysis.py
python 06_internal_validation_ids.py
python 07_external_validation_prep.py
```

Альтернатива — Google Colab: `notebooks/full_pipeline_colab.py`.

## Источники данных

| Проект PRIDE | n образцов | Тип материала | Cancer / Control |
|---|---|---|---|
| PXD033741 | 110 | Ткань | 80 / 30 |
| PXD009655 | 50 | Микровезикулы | 43 / 7 |
| PXD012998 | 14 | Ткань | 10 / 4 |
| PXD025864 | 6 | Ткань | 3 / 3 |
| **Итого** | **180** | — | **136 / 44** |

Объединённая матрица: 9 610 белков × 180 образцов, z-score-нормализована внутри каждого проекта без импутации пропусков.

## Контакты

ИБМХ
