# Входные данные

В этой папке должен находиться файл:

```
matrix_after_project_zscore_no_imputation.xlsx
```

Размер: ~15 МБ, 2 листа (`matrix_after_project_z` и `sample_annotation`).

## Что внутри файла

### Лист 1: `matrix_after_project_z`

Объединённая матрица интенсивностей белков:
- Колонки 1–4: метаданные (`Protein IDs`, `Gene names`, `Protein names`, `Majority protein IDs`)
- Колонки 5–184: 180 образцов из 4 проектов PRIDE
- 9 610 строк (белков)
- Значения: log2-преобразованные iBAQ интенсивности, **z-score-нормализованы внутри каждого проекта** (без импутации пропусков)

### Лист 2: `sample_annotation`

Аннотация образцов (180 строк):
- `SampleColumn` — имя колонки в матрице
- `Project` — PXD033741 / PXD009655 / PXD012998 / PXD025864
- `Status` — Cancer / Control

## Источники

Исходные raw-данные публичны и доступны через PRIDE:

| Проект | Ссылка | Тип материала |
|---|---|---|
| PXD033741 | https://www.ebi.ac.uk/pride/archive/projects/PXD033741 | Ткань |
| PXD009655 | https://www.ebi.ac.uk/pride/archive/projects/PXD009655 | Микровезикулы |
| PXD012998 | https://www.ebi.ac.uk/pride/archive/projects/PXD012998 | Ткань |
| PXD025864 | https://www.ebi.ac.uk/pride/archive/projects/PXD025864 | Ткань |

## Воспроизведение объединённой матрицы
- белки в строках, образцы в колонках
- метаданные в первых 4 колонках (`Protein IDs`, `Gene names`, `Protein names`, `Majority protein IDs`)
- z-score-нормализованные значения, NaN для непродетектированных белков
- лист `sample_annotation` с колонками `SampleColumn`, `Project`, `Status`
