# Списки белков

В этой папке сохраняются финальные списки белков с UniProt IDs, gene names и метриками.

## Создаются скриптом 04 (`04_protein_lists_export.py`)

| Файл | Содержание |
|---|---|
| `all_core_proteins_with_metrics.csv` | Все 2701 коровых белков с RF importance, t-stats, hallmark |
| `top100_RF_proteins.csv` | Топ-100 по Random Forest importance |
| `universal_markers_tissue_EVs.csv` | 194 universal markers (значимые в обоих матриксах) |
| `PROTEIN_LIST_FULL.xlsx` | Excel со всеми листами |

## Создаются скриптом 05 (`05_direction_analysis.py`)

| Файл | Содержание |
|---|---|
| `top100_RF_with_direction.csv` | Top-100 с разметкой ↑/↓ в раке |
| `universal_with_direction.csv` | Universal markers с разметкой |
| `intersection_with_direction.csv` | 9 белков пересечения (особо ценные) |
| `all_DOWNregulated_in_cancer.csv` | Все 269 значимо пониженных в раке |
| `PROTEINS_BY_DIRECTION.xlsx` | Excel с разбиением по направлениям |

## Создаются скриптом 06 (`06_internal_validation_ids.py`)

| Файл | Содержание |
|---|---|
| `internal_validation_report.txt` | Отчёт о внутренней проверке |
| `all_proteins_with_validation_flags.csv` | Все белки с флагами валидации |
| `duplicate_gene_names.csv` | Дубликаты gene names (изоформы) |
| `likely_unreviewed_TrEMBL.csv` | Эвристически TrEMBL-записи |

## Создаются скриптом 07 (`07_external_validation_prep.py`)

| Файл | Содержание |
|---|---|
| `uniprot_ids_TOP100.txt` | Список UniProt IDs топ-100 (для UniProt batch) |
| `uniprot_ids_UNIVERSAL.txt` | Список UniProt IDs universal markers |
| `uniprot_ids_all.txt` | Все UniProt IDs |
| `gene_names_TOP100.txt` | Gene names топ-100 (для Enrichr, STRING) |
| `gene_names_UNIVERSAL.txt` | Gene names universal markers |
| `gene_names_all.txt` | Все gene names |
| `README_external_validation.txt` | Пошаговая инструкция по внешней проверке |

## Соотношение списков

```
Top-100 RF (n=100, только ткань)
  ├── 91 только в Top-100
  └── 9 пересекаются с Universal

Universal markers (n=194, ткань + EVs)
  ├── 185 только в Universal
  └── 9 пересекаются с Top-100

Объединение всех уникальных: 285 белков
```

Пересекающиеся 9 белков: **SEC11A, TMCO1, ACAA1, DTX3L, THEM6, MTA2, IL4I1, OCIAD2, SHMT2** — наиболее ценные кандидаты в биомаркеры жидкой биопсии (работают и в ткани, и в EVs).
