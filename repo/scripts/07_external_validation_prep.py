"""
Скрипт 7: Подготовка файлов для внешней валидации белков
=========================================================

ВХОД:  results/protein_lists/all_core_proteins_with_metrics.csv
       results/protein_lists/top100_RF_proteins.csv
       results/protein_lists/universal_markers_tissue_EVs.csv
ВЫХОД: results/protein_lists/uniprot_ids_TOP100.txt
       results/protein_lists/uniprot_ids_UNIVERSAL.txt
       results/protein_lists/uniprot_ids_all.txt
       results/protein_lists/gene_names_TOP100.txt
       results/protein_lists/gene_names_UNIVERSAL.txt
       results/protein_lists/gene_names_all.txt
       results/protein_lists/README_external_validation.txt

Что делает:
  Готовит файлы для batch-загрузки в:
    • UniProt ID Mapping (https://www.uniprot.org/id-mapping)
    • Enrichr (https://maayanlab.cloud/Enrichr/)
    • STRING-DB (https://string-db.org/)
    • DisGeNET (https://www.disgenet.org/)
    • Human Protein Atlas (https://www.proteinatlas.org/)
"""

import os
import pandas as pd

# ─────────────────────────────────────────────────────────
INPUT = '../results/protein_lists/all_core_proteins_with_metrics.csv'
TOP100 = '../results/protein_lists/top100_RF_proteins.csv'
UNIVERSAL = '../results/protein_lists/universal_markers_tissue_EVs.csv'
OUTPUT_DIR = '../results/protein_lists'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)
ids_all = df['UniProt_ID'].dropna().unique().tolist()
with open(f'{OUTPUT_DIR}/uniprot_ids_all.txt', 'w') as f:
    f.write('\n'.join(ids_all))

genes = df['Gene_name'].dropna().astype(str).str.strip()
genes = genes[genes != ''].unique().tolist()
with open(f'{OUTPUT_DIR}/gene_names_all.txt', 'w') as f:
    f.write('\n'.join(genes))
print(f"All proteins: {len(ids_all)} IDs, {len(genes)} genes")

top100 = pd.read_csv(TOP100)
ids_top = top100['UniProt_ID'].dropna().tolist()
with open(f'{OUTPUT_DIR}/uniprot_ids_TOP100.txt', 'w') as f:
    f.write('\n'.join(ids_top))
genes_top = top100['Gene_name'].dropna().astype(str).str.strip().tolist()
with open(f'{OUTPUT_DIR}/gene_names_TOP100.txt', 'w') as f:
    f.write('\n'.join(genes_top))
print(f"Top-100: {len(ids_top)} IDs, {len(genes_top)} genes")

uni = pd.read_csv(UNIVERSAL)
ids_uni = uni['UniProt_ID'].dropna().tolist()
with open(f'{OUTPUT_DIR}/uniprot_ids_UNIVERSAL.txt', 'w') as f:
    f.write('\n'.join(ids_uni))
genes_uni = uni['Gene_name'].dropna().astype(str).str.strip().tolist()
with open(f'{OUTPUT_DIR}/gene_names_UNIVERSAL.txt', 'w') as f:
    f.write('\n'.join(genes_uni))
print(f"Universal: {len(ids_uni)} IDs, {len(genes_uni)} genes")


readme = """ИНСТРУКЦИЯ ПО ВНЕШНЕЙ ВАЛИДАЦИИ БЕЛКОВ


Файлы в этой папке:
  uniprot_ids_TOP100.txt    — топ-100 RF importance (приоритет проверки)
  uniprot_ids_UNIVERSAL.txt — universal markers (для PPI-графа)
  uniprot_ids_all.txt       — все коровые белки
  gene_names_*.txt          — те же списки в виде gene names

────────────────────────────────────────────────────────────────────
ШАГ 1. UniProt ID Mapping — актуальная проверка идентификаторов
────────────────────────────────────────────────────────────────────
URL: https://www.uniprot.org/id-mapping

1. Открыть uniprot_ids_TOP100.txt, скопировать всё содержимое.
2. На сайте: From — UniProtKB AC/ID, To — UniProtKB.
3. Вставить и нажать Map IDs.
4. Скачать TSV/Excel.

Покажет:
  • актуальные/secondary/deprecated идентификаторы
  • Reviewed (Swiss-Prot) vs Unreviewed (TrEMBL)
  • актуальные gene names, protein names
  • subcellular location, GO-аннотации

────────────────────────────────────────────────────────────────────
ШАГ 2. Enrichr — функциональное обогащение
────────────────────────────────────────────────────────────────────
URL: https://maayanlab.cloud/Enrichr/

1. Открыть gene_names_TOP100.txt
2. Скопировать список и вставить в текстовое поле на сайте
3. Submit

Категории для рака яичников:
  • Pathways → MSigDB Hallmark 2020
  • Pathways → KEGG 2021 Human
  • Pathways → Reactome 2022
  • Diseases/Drugs → DisGeNET, OMIM Disease

────────────────────────────────────────────────────────────────────
ШАГ 3. STRING — PPI-граф для планового результата 2025-2026
────────────────────────────────────────────────────────────────────
URL: https://string-db.org/cgi/input

1. Multiple proteins → вставить gene_names_UNIVERSAL.txt
2. Organism: Homo sapiens
3. Settings:
   • Confidence: medium (0.4) или high (0.7)
   • 1st shell: 10–20 interactors (для ≥25 кандидатов)
4. Скачать:
   • Network image (PNG/SVG)
   • TSV таблицу взаимодействий
   • Functional enrichment

────────────────────────────────────────────────────────────────────
ШАГ 4. DisGeNET — связь с раком яичников
────────────────────────────────────────────────────────────────────
URL: https://www.disgenet.org/search

Поиск по gene_names_TOP100.txt с ассоциациями:
  • C0029925 (Ovarian Carcinoma)
  • C0007115 (Breast Carcinoma) — для сравнения

────────────────────────────────────────────────────────────────────
ШАГ 5. Human Protein Atlas — subcellular localization
────────────────────────────────────────────────────────────────────
URL: https://www.proteinatlas.org/

Важно для биомаркеров:
  • Secreted белки — подходят для плазмы/сыворотки
  • Мембранные — могут быть в EVs
  • Ядерные — только в ткани, не в жидкой биопсии

────────────────────────────────────────────────────────────────────
ФОРМАТЫ ФАЙЛОВ
────────────────────────────────────────────────────────────────────
uniprot_ids_*.txt — по одному ID на строку (стандарт)
gene_names_*.txt  — gene symbols (для Enrichr, DAVID)

Все файлы в UTF-8.
"""

with open(f'{OUTPUT_DIR}/README_external_validation.txt', 'w', encoding='utf-8') as f:
    f.write(readme)

