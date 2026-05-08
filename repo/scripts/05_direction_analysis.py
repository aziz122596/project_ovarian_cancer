"""
Скрипт 5: Анализ направления изменений (повышены /понижены в раке)
=======================================================================

ВХОД:  results/protein_lists/all_core_proteins_with_metrics.csv
       (создаётся скриптом 04)
ВЫХОД: results/protein_lists/top100_RF_with_direction.csv
       results/protein_lists/universal_with_direction.csv
       results/protein_lists/intersection_with_direction.csv
       results/protein_lists/all_DOWNregulated_in_cancer.csv
       results/protein_lists/PROTEINS_BY_DIRECTION.xlsx

Что делает:
  1. Размечает белки по направлению изменения в раке (на основе t-stat)
  2. Считает соотношение повышенных vs пониженных в каждом списке
  3. Отдельно сохраняет список пониженных белков (для анализа стромы/ECM)
  4. Создаёт Excel с разбиением по направлениям
"""

import os
import pandas as pd

INPUT = '../results/protein_lists/all_core_proteins_with_metrics.csv'
TOP100 = '../results/protein_lists/top100_RF_proteins.csv'
UNIVERSAL = '../results/protein_lists/universal_markers_tissue_EVs.csv'
OUTPUT_DIR = '../results/protein_lists'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def direction_label(t):
    if pd.isna(t) or t == 0:
        return '—'
    return '↑ UP in Cancer' if t > 0 else '↓ DOWN in Cancer'

all_data = pd.read_csv(INPUT)
top100 = pd.read_csv(TOP100)
universal = pd.read_csv(UNIVERSAL)

top100_ids = set(top100['UniProt_ID'])
universal_ids = set(universal['UniProt_ID'])
intersection_ids = top100_ids & universal_ids

top100['Direction_Tissue'] = top100['t_stat_Tissue'].apply(direction_label)
top100['Direction_EVs'] = top100['t_stat_EVs'].apply(direction_label)
top100['Same_direction'] = (top100['t_stat_Tissue'] > 0) == (top100['t_stat_EVs'] > 0)

n_up_T = (top100['t_stat_Tissue'] > 0).sum()
n_down_T = (top100['t_stat_Tissue'] < 0).sum()

top100[['Rank_RF', 'Gene_name', 'UniProt_ID', 'Protein_name', 'RF_importance',
        'Direction_Tissue', 'Direction_EVs', 'Same_direction',
        't_stat_Tissue', 'p_value_Tissue', 't_stat_EVs', 'p_value_EVs',
        'Hallmark_category']].to_csv(
    f'{OUTPUT_DIR}/top100_RF_with_direction.csv', index=False, encoding='utf-8-sig'
)

universal['Direction'] = universal['t_stat_Tissue'].apply(direction_label)
n_up_U = (universal['t_stat_Tissue'] > 0).sum()
n_down_U = (universal['t_stat_Tissue'] < 0).sum()
print(f"\nUNIVERSAL MARKERS (n={len(universal)}):")
print(f"  ↑ UP:   {n_up_U}")
print(f"  ↓ DOWN: {n_down_U}")
universal.to_csv(f'{OUTPUT_DIR}/universal_with_direction.csv',
                 index=False, encoding='utf-8-sig')

inter_df = all_data[all_data['UniProt_ID'].isin(intersection_ids)].copy()
inter_df['Direction_Tissue'] = inter_df['t_stat_Tissue'].apply(direction_label)
inter_df['Direction_EVs'] = inter_df['t_stat_EVs'].apply(direction_label)
inter_df = inter_df.sort_values('Rank_RF')

print(f"\nПЕРЕСЕЧЕНИЕ (n={len(inter_df)}):")
print(inter_df[['Gene_name', 'UniProt_ID', 'Direction_Tissue', 'Direction_EVs',
                't_stat_Tissue', 't_stat_EVs']].to_string(index=False))
inter_df.to_csv(f'{OUTPUT_DIR}/intersection_with_direction.csv',
                index=False, encoding='utf-8-sig')

strong_down = all_data[
    (all_data['t_stat_Tissue'] < -3) &
    (all_data['p_value_Tissue'] < 0.05)
].copy().sort_values('t_stat_Tissue')

strong_up = all_data[
    (all_data['t_stat_Tissue'] > 3) &
    (all_data['p_value_Tissue'] < 0.05)
].copy()

strong_down.to_csv(f'{OUTPUT_DIR}/all_DOWNregulated_in_cancer.csv',
                    index=False, encoding='utf-8-sig')


with pd.ExcelWriter(f'{OUTPUT_DIR}/PROTEINS_BY_DIRECTION.xlsx', engine='openpyxl') as writer:
    summary = pd.DataFrame({
        'Список': [
            'TOP-100 RF: ВСЕГО',
            '  ↑ повышены (ткань)',
            '  ↓ понижены (ткань)',
            '',
            'Universal markers: ВСЕГО',
            '  ↑ повышены в обоих',
            '  ↓ понижены в обоих',
            '',
            'Все коровые: ВСЕГО',
            f'  ↑ значимо повышены (|t|>3)',
            f'  ↓ значимо понижены (|t|>3)',
        ],
        'Количество': [
            len(top100), n_up_T, n_down_T, '',
            len(universal), n_up_U, n_down_U, '',
            len(all_data), len(strong_up), len(strong_down),
        ],
    })
    summary.to_excel(writer, sheet_name='0. Сводка', index=False)

    t_up = top100[top100['Direction_Tissue'] == '↑ UP in Cancer']
    t_dn = top100[top100['Direction_Tissue'] == '↓ DOWN in Cancer']
    t_up.to_excel(writer, sheet_name=f'1. TOP100 UP (n={len(t_up)})', index=False)
    if len(t_dn) > 0:
        t_dn.to_excel(writer, sheet_name=f'2. TOP100 DOWN (n={len(t_dn)})', index=False)

    u_up = universal[universal['Direction'] == '↑ UP in Cancer']
    u_dn = universal[universal['Direction'] == '↓ DOWN in Cancer']
    u_up.to_excel(writer, sheet_name=f'3. Univ UP (n={len(u_up)})', index=False)
    u_dn.to_excel(writer, sheet_name=f'4. Univ DOWN (n={len(u_dn)})', index=False)

    inter_df.to_excel(writer, sheet_name=f'5. Intersection (n={len(inter_df)})', index=False)
    strong_down.head(50).to_excel(writer, sheet_name='6. Top-50 DOWN ECM-like', index=False)
