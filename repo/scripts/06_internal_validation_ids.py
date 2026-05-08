"""
Скрипт 6: Внутренняя валидация белковых идентификаторов
========================================================

ВХОД:  results/protein_lists/all_core_proteins_with_metrics.csv
       (создаётся скриптом 04)
ВЫХОД: results/protein_lists/internal_validation_report.txt
       results/protein_lists/all_proteins_with_validation_flags.csv
       results/protein_lists/duplicate_gene_names.csv
       results/protein_lists/likely_unreviewed_TrEMBL.csv

Что делает:
  1. Проверяет формат UniProt IDs (regex)
  2. Ищет пустые/missing идентификаторы
  3. Находит дубликаты
  4. Эвристически разделяет Swiss-Prot vs TrEMBL
  5. Sanity-check на 17 известных human-белках
"""

import re
import os
import pandas as pd
import numpy as np

INPUT = '../results/protein_lists/all_core_proteins_with_metrics.csv'
OUTPUT_DIR = '../results/protein_lists'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)
print(f"Всего белков: {len(df)}")

UNIPROT_RE = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|'
    r'^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
)
valid_format = df['UniProt_ID'].astype(str).apply(lambda x: bool(UNIPROT_RE.match(x)))

empty_uniprot = df['UniProt_ID'].isna() | (df['UniProt_ID'].astype(str).str.strip() == '')
empty_gene = df['Gene_name'].isna() | (df['Gene_name'].astype(str).str.strip() == '')
empty_name = df['Protein_name'].isna() | (df['Protein_name'].astype(str).str.strip() == '')

dup_uniprot = df['UniProt_ID'].duplicated(keep=False)
dup_gene = df['Gene_name'].duplicated(keep=False) & ~empty_gene
print(f"\n3. Дубликаты:")
print(f"   UniProt:    {dup_uniprot.sum()}")
print(f"   Gene name:  {dup_gene.sum()}")

if dup_gene.sum() > 0:
    df[dup_gene][['Gene_name', 'UniProt_ID', 'Protein_name']].to_csv(
        f'{OUTPUT_DIR}/duplicate_gene_names.csv', index=False, encoding='utf-8-sig'
    )

def likely_swiss_prot(uid):
    if pd.isna(uid):
        return False
    s = str(uid).strip()
    if len(s) == 6 and s[0] in 'OPQ':
        return True
    if len(s) == 6 and s[0] in 'ABCDEFGHIJKLMNRSTUVWXYZ' and s[1].isdigit():
        return True
    return False

df['likely_reviewed'] = df['UniProt_ID'].apply(likely_swiss_prot)
n_reviewed = df['likely_reviewed'].sum()
n_unreviewed = (~df['likely_reviewed']).sum()
print(f"\n4. Эвристика Swiss-Prot vs TrEMBL:")
print(f"   Swiss-Prot (reviewed):  {n_reviewed} ({n_reviewed/len(df)*100:.1f}%)")
print(f"   TrEMBL (unreviewed):    {n_unreviewed} ({n_unreviewed/len(df)*100:.1f}%)")

if n_unreviewed > 0:
    df[~df['likely_reviewed']][['Gene_name', 'UniProt_ID', 'Protein_name']].to_csv(
        f'{OUTPUT_DIR}/likely_unreviewed_TrEMBL.csv', index=False, encoding='utf-8-sig'
    )

KNOWN = {
    'GAPDH': 'P04406', 'ACTB': 'P60709', 'TUBA1B': 'P68363',
    'HSPA8': 'P11142', 'HSP90AA1': 'P07900', 'TP53': 'P04637',
    'BAX': 'Q07812', 'AIFM1': 'O95831', 'CS': 'O75390',
    'IDH2': 'P48735', 'COX4I1': 'P13073', 'EIF2S1': 'P05198',
    'SEC11A': 'P67812', 'COPA': 'P53621', 'COPB1': 'P53618',
    'COPB2': 'P35606', 'TMCO1': 'Q9UM00',
}
sanity_ok = sanity_mismatch = sanity_missing = 0
for gene, expected in KNOWN.items():
    found = df[df['Gene_name'] == gene]
    if found.empty:
        sanity_missing += 1
    else:
        actual = found['UniProt_ID'].iloc[0]
        if actual == expected:
            sanity_ok += 1
        else:
            all_ids = str(found['UniProt_All_IDs'].iloc[0]).split(';')
            if expected in all_ids:
                sanity_ok += 1
            else:
                sanity_mismatch += 1


report = f"""ВНУТРЕННЯЯ ВАЛИДАЦИЯ ИДЕНТИФИКАТОРОВ

Всего проверено белков: {len(df)}

ФОРМАТ
  Валидный формат:      {valid_format.sum()} ({valid_format.mean()*100:.2f}%)
  Битый формат:         {(~valid_format).sum()}

ПОЛНОТА
  Пустых UniProt:       {empty_uniprot.sum()}
  Пустых Gene_name:     {empty_gene.sum()}
  Пустых Protein_name:  {empty_name.sum()}

УНИКАЛЬНОСТЬ
  Дубликатов UniProt:   {dup_uniprot.sum()}
  Дубликатов Gene_name: {dup_gene.sum()}

КАЧЕСТВО ИСТОЧНИКА
  Swiss-Prot:           {n_reviewed} ({n_reviewed/len(df)*100:.1f}%)
  TrEMBL:               {n_unreviewed} ({n_unreviewed/len(df)*100:.1f}%)

SANITY CHECK ({len(KNOWN)} известных белков)
  Совпало:              {sanity_ok}
  Не совпало:           {sanity_mismatch}
  Нет в выборке:        {sanity_missing}

ВЕРДИКТ
"""
if (~valid_format).sum() == 0:
    report += " Все идентификаторы валидны.\n"
if dup_uniprot.sum() == 0:
    report += " Дубликатов UniProt нет.\n"
if sanity_mismatch == 0:
    report += " Sanity check пройден.\n"
if n_reviewed / len(df) > 0.95:
    report += f" {n_reviewed/len(df)*100:.1f}% — Swiss-Prot (reviewed).\n"

print("\n" + report)
with open(f'{OUTPUT_DIR}/internal_validation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

df['format_valid'] = valid_format
df['is_duplicate_uniprot'] = dup_uniprot
df.to_csv(f'{OUTPUT_DIR}/all_proteins_with_validation_flags.csv',
          index=False, encoding='utf-8-sig')
