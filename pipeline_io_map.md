# Pipeline I/O map

Scripts scanned: 64

## Per script

### 01_parse_xml.py
- PATH? `words_labeled.csv` (line 34)
- PATH? `.terminals.xml` (line 61)
- PATH? `.xml` (line 71)
- PATH? `.terminals.xml` (line 76)
- PATH? `.xml` (line 147)

### 02_lexical_surprisal.py
- PATH? `clean_word_index.csv` (line 36)
- PATH? `pipeline_log.txt` (line 37)
- PATH? `lexical_surprisal.csv` (line 38)
- PATH? `.xml` (line 41)
- PATH? `.txt` (line 68)
- PATH? `  Using sentence boundaries from clean_word_index.csv` (line 136)
- PATH? `Sentence boundaries: from clean_word_index.csv` (line 174)

### 03_syntactic_surprisal.py
- PATH? `words_labeled.csv` (line 20)
- PATH? `syntactic_surprisal.csv` (line 21)

### 03_syntactic_surprisal_modelblocks.py
- PATH? `clean_word_index.csv` (line 36)
- PATH? `syntactic_surprisal.csv` (line 38)
- PATH? `pipeline_log.txt` (line 39)

### 03a_clean_corpus.py
- PATH? `words_labeled.csv` (line 37)
- PATH? `clean_sentences.txt` (line 38)
- PATH? `clean_word_index.csv` (line 39)
- PATH? `pipeline_log.txt` (line 40)
- PATH? `{conv_id}.{speaker}.disfluency.xml` (line 79)

### 03b_parallel_modelblocks.py
- PATH? `pipeline_log.txt` (line 33)
- PATH? `genmodel/testing/chunk_{ci:02d}.tokdecs.log` (line 120)
- PATH? `genmodel/testing/chunk_{ci:02d}.tokdecs.log` (line 148)

### 03b_prepare_modelblocks.py
- PATH? `clean_sentences.txt` (line 31)
- PATH? `clean_word_index.csv` (line 32)
- PATH? `clean_word_index.csv` (line 35)
- PATH? `pipeline_log.txt` (line 36)
- PATH? `Input: clean_sentences.txt` (line 57)
- PATH? `  2> genmodel/testing/clean_sentences.tokdecs.log` (line 120)

### 04_dlt_costs.py
- PATH? `clean_word_index.csv` (line 42)
- PATH? `dlt_costs.csv` (line 43)
- PATH? `pipeline_log.txt` (line 44)
- PATH? `Sentence boundaries: from clean_word_index.csv` (line 161)

### 04b_merge_modelblocks.py
- PATH? `words_labeled.csv` (line 22)
- PATH? `syntactic_surprisal_modelblocks.csv` (line 31)
- PATH? `syntactic_surprisal.csv` (line 33)

### 05_build_dataset.py
- PATH? `words_labeled.csv` (line 45)
- PATH? `lexical_surprisal.csv` (line 46)
- PATH? `dlt_costs.csv` (line 47)
- PATH? `syntactic_surprisal.csv` (line 48)
- PATH? `words_features.csv` (line 49)
- PATH? `dataset_filler.csv` (line 50)
- PATH? `dataset_reparandum.csv` (line 51)
- PATH? `pipeline_log.txt` (line 52)

### 05_build_dataset_v2.py
- PATH? `words_features.csv` (line 53)
- PATH? `words_labeled.csv` (line 54)
- PATH? `dataset_filler_v2.csv` (line 55)
- PATH? `build_dataset_v2_log.txt` (line 56)

### 06_run_glmm.R
- PATH? `dataset_filler.csv` (line 23)
- PATH? `dataset_reparandum.csv` (line 24)
- WRITE `.csv` (line 145)
- PATH? `.txt` (line 154)
- WRITE `table2_random_effects.csv` (line 221)

### 06_run_glmm_filler_v2.R
- PATH? `dataset_filler_v2.csv` (line 27)
- WRITE `table1_filler_v2.csv` (line 145)
- PATH? `accuracy_filler_v2.txt` (line 153)

### 06_run_glmm_filler_v2_table2.R
- PATH? `dataset_filler_v2.csv` (line 33)
- WRITE `table2_filler_v2.csv` (line 178)

### 07_parse_phonwords.py
- PATH? `phonwords_index.csv` (line 58)
- PATH? `pipeline_log.txt` (line 59)
- PATH? `.phonwords.xml` (line 94)
- PATH? `.phonwords.xml` (line 103)

### 07_table3.py
- PATH? `words_features.csv` (line 20)
- PATH? `table3_elongation.csv` (line 21)
- PATH? `table3_elongation.txt` (line 22)

### 08_bridge_terminals_phonwords.py
- PATH? `words_labeled.csv` (line 60)
- PATH? `phonwords_index.csv` (line 61)
- PATH? `accents_index.csv` (line 62)
- PATH? `breaks_index.csv` (line 63)
- PATH? `word_prosody_bridge.csv` (line 64)
- PATH? `pipeline_log.txt` (line 65)
- PATH? `.xml` (line 110)
- PATH? `.terminals.xml` (line 115)

### 08_figures.py
- PATH? `words_features.csv` (line 16)
- PATH? `figure1_elongation_by_label.png` (line 43)
- PATH? `figure2_function_words.png` (line 94)
- PATH? `figure3_surprisal_by_label.png` (line 125)
- PATH? `figure4_dlt_by_label.png` (line 169)
- PATH? `  results/figure1_elongation_by_label.png` (line 195)
- PATH? `  results/figure2_function_words.png` (line 196)
- PATH? `  results/figure3_surprisal_by_label.png` (line 197)
- PATH? `  results/figure4_dlt_by_label.png` (line 198)

### 09_parse_accents.py
- PATH? `accents_index.csv` (line 49)
- PATH? `pipeline_log.txt` (line 50)
- PATH? `.accents.xml` (line 96)
- PATH? `.accents.xml` (line 104)

### 09_pos_analysis.py
- PATH? `words_features.csv` (line 18)
- PATH? `paper_figure1_pos_intercepts.png` (line 130)
- PATH? `paper_figure2_storage_cost_pos.png` (line 210)
- PATH? `  results/paper_figure1_pos_intercepts.png` (line 232)
- PATH? `  results/paper_figure2_storage_cost_pos.png` (line 233)

### 10_compare_results.py
- PATH? `accuracy_filler.txt` (line 45)
- PATH? `accuracy_reparandum.txt` (line 46)
- PATH? `accuracy_filler_modelblocks.txt` (line 49)
- PATH? `accuracy_reparandum_modelblocks.txt` (line 50)
- PATH? `table1_reparandum.csv` (line 64)
- PATH? `table1_reparandum_modelblocks.csv` (line 66)
- PATH? `figure_pcfg_vs_modelblocks.png` (line 199)

### 10_parse_breaks.py
- PATH? `breaks_index.csv` (line 60)
- PATH? `pipeline_log.txt` (line 61)
- PATH? `.breaks.xml` (line 102)
- PATH? `.breaks.xml` (line 110)

### 11_build_prosodic_features.py
- PATH? `word_prosody_bridge.csv` (line 59)
- PATH? `clean_word_index.csv` (line 60)
- PATH? `words_labeled.csv` (line 61)
- PATH? `prosodic_features.csv` (line 62)
- PATH? `pipeline_log.txt` (line 63)
- PATH? `dataset_filler_v3.csv and dataset_reparandum_v3.csv` (line 244)

### 12_merge_prosodic_dataset.py
- PATH? `dataset_filler_v2.csv` (line 76)
- PATH? `dataset_reparandum.csv` (line 77)
- PATH? `prosodic_features.csv` (line 78)
- PATH? `dataset_filler_accent.csv` (line 80)
- PATH? `dataset_filler_break.csv` (line 81)
- PATH? `dataset_reparandum_accent.csv` (line 82)
- PATH? `dataset_reparandum_break.csv` (line 83)
- PATH? `pipeline_log.txt` (line 85)

### 13_run_glmm_prosodic.R
- PATH? `pipeline_log.txt` (line 47)
- PATH? `dataset_filler_accent.csv` (line 222)
- PATH? `dataset_filler_break.csv` (line 229)
- PATH? `dataset_reparandum_accent.csv` (line 236)
- PATH? `dataset_reparandum_break.csv` (line 243)
- PATH? `table_prosodic_comparison.csv` (line 267)

### 14_gpt2_surprisal.py
- PATH? `clean_word_index.csv` (line 44)
- PATH? `gpt2_surprisal.csv` (line 45)
- PATH? `pipeline_log.txt` (line 46)

### 15_merge_gpt2_dataset.py
- PATH? `gpt2_surprisal.csv` (line 54)
- PATH? `words_labeled.csv` (line 55)
- PATH? `dataset_filler_v2.csv` (line 56)
- PATH? `dataset_reparandum.csv` (line 57)
- PATH? `gpt2_surprisal_propagated.csv` (line 59)
- PATH? `dataset_filler_gpt2.csv` (line 60)
- PATH? `dataset_reparandum_gpt2.csv` (line 61)
- PATH? `pipeline_log.txt` (line 63)

### 16_run_glmm_gpt2.R
- PATH? `pipeline_log.txt` (line 53)
- PATH? `dataset_filler_gpt2.csv` (line 222)
- PATH? `dataset_reparandum_gpt2.csv` (line 229)
- PATH? `table_gpt2_comparison.csv` (line 253)

### 17_consolidate_results.py
- PATH? `pipeline_log.txt` (line 38)
- PATH? `table_prosodic_comparison.csv` (line 67)
- PATH? `table_gpt2_comparison.csv` (line 85)
- PATH? `table_all_features_comparison.csv` (line 106)

### 17_multi_llm_surprisal.py
- PATH? `clean_word_index.csv` (line 69)
- PATH? `pipeline_log.txt` (line 70)
- PATH? `multi_llm_run_summary.csv` (line 71)
- PATH? `surprisal_{tag}.csv` (line 179)
- PATH? `\nNext: python src/18_merge_multi_llm.py   (merges each surprisal_<tag>.csv` (line 322)

### 18_merge_multi_llm.py
- PATH? `dataset_filler_v2.csv` (line 56)
- PATH? `dataset_reparandum.csv` (line 57)
- PATH? `dataset_filler_multillm.csv` (line 58)
- PATH? `dataset_reparandum_multillm.csv` (line 59)
- PATH? `surprisal_{tag}.csv` (line 136)

### 18_merge_multi_llm_FIXED_v2.py
- PATH? `dataset_filler_v2.csv` (line 56)
- PATH? `dataset_reparandum.csv` (line 57)
- PATH? `dataset_filler_multillm.csv` (line 58)
- PATH? `dataset_reparandum_multillm.csv` (line 59)
- PATH? `surprisal_{tag}.csv` (line 142)

### 19_run_glmm_multi_llm.R
- PATH? `dataset_filler_multillm.csv` (line 45)
- PATH? `dataset_reparandum_multillm.csv` (line 46)
- PATH? `table_multi_llm_comparison.csv` (line 214)

### 19_run_glmm_multi_llm_FIXED_v2.R
- PATH? `dataset_filler_multillm.csv` (line 45)
- PATH? `dataset_reparandum_multillm.csv` (line 46)
- PATH? `table_multi_llm_comparison.csv` (line 222)
- PATH? `_rerun.csv` (line 223)

### 20_plot_multi_llm_comparison.py
- READ `results/table_multi_llm_comparison.csv` (line 20)
- WRITE `results/multi_llm_comparison.png` (line 106)
- WRITE `Saved -> results/multi_llm_comparison.png` (line 107)

### build_short_index.py
- PATH? `words_labeled.csv` (line 18)
- PATH? `swbd_word_index.csv` (line 19)

### check_deps.py
- no data file paths found

### check_filler_join.py
- READ `data/processed/dataset_filler_v2.csv` (line 3)
- READ `data/processed/surprisal_gpt2.csv` (line 4)

### check_fixed_merge.py
- READ `data/processed/dataset_filler_v2.csv` (line 4)
- READ `data/processed/surprisal_gpt2.csv` (line 5)

### check_foll_imputation.py
- READ `data/processed/surprisal_gpt2.csv` (line 19)
- READ `data/processed/dataset_filler_multillm.csv` (line 20)
- READ `data/processed/dataset_filler_gpt2.csv` (line 21)
- WRITE `foll_imputation_check.txt` (line 44)
- WRITE `\nsaved to foll_imputation_check.txt` (line 46)

### check_foll_semantics.py
- READ `data/processed/surprisal_gpt2.csv` (line 17)
- PATH? `data/processed/dataset_filler_gpt2.csv` (line 56)
- PATH? `data/processed/dataset_filler_multillm.csv` (line 57)
- WRITE `foll_semantics_check.txt` (line 59)
- WRITE `\nsaved to foll_semantics_check.txt` (line 61)

### check_na_pattern.py
- READ `data/processed/dataset_filler_v2.csv` (line 3)

### check_na_rate.py
- READ `data/processed/dataset_filler_v2.csv` (line 3)

### check_reparandum_merge.py
- READ `data/processed/surprisal_gpt2.csv` (line 18)
- PATH? `data/processed/dataset_reparandum_gpt2.csv` (line 48)
- PATH? `data/processed/dataset_reparandum_multillm.csv` (line 49)
- READ+WRITE? `reparandum_merge_check.txt` (line 51)
- WRITE `\nsaved to reparandum_merge_check.txt` (line 53)

### check_row_sets.py
- READ `data/processed/dataset_filler_gpt2.csv` (line 19)
- READ `data/processed/dataset_filler_multillm.csv` (line 20)
- READ+WRITE? `row_sets_check.txt` (line 62)
- WRITE `\nsaved to row_sets_check.txt` (line 64)

### check_sentences.py
- READ `data/processed/words_labeled.csv` (line 10)
- READ `data/processed/clean_word_index.csv` (line 11)

### check_word_column.py
- PATH? `data/processed/dataset_filler_gpt2.csv` (line 18)
- PATH? `data/processed/dataset_filler_accent.csv` (line 19)
- PATH? `data/processed/dataset_filler_multillm.csv` (line 20)
- WRITE `word_column_check.txt` (line 54)
- WRITE `\nsaved to word_column_check.txt` (line 56)

### compare_pcfg_modelblocks.py
- READ `syntactic_surprisal.csv` (line 27)
- READ `swbd_5000_surprisal.csv` (line 31)
- WRITE `pcfg_vs_modelblocks_comparison.csv` (line 89)
- PATH? `figure_pcfg_vs_modelblocks.png` (line 125)
- PATH? `  results/pcfg_vs_modelblocks_comparison.csv` (line 134)
- PATH? `  results/figure_pcfg_vs_modelblocks.png` (line 135)

### convert_swbd_syntax.py
- PATH? `.xml` (line 43)
- PATH? `.terminals.xml` (line 48)
- PATH? `.syntax.xml` (line 129)
- PATH? `.xml` (line 161)

### diagnose_filler_dropped_rows.py
- PATH? `dataset_filler_v2.csv` (line 40)
- PATH? `words_labeled.csv` (line 41)
- PATH? `gpt2_surprisal_propagated.csv` (line 42)

### diagnose_filler_dropped_rows_v2.py
- PATH? `dataset_filler_v2.csv` (line 32)
- PATH? `gpt2_surprisal_propagated.csv` (line 33)

### diagnose_modelblocks_failures.py
- PATH? `clean_word_index.csv` (line 40)
- PATH? `syntactic_surprisal.csv` (line 41)

### diagnose_v3.py
- PATH? `dataset_filler_v2.csv` (line 26)

### diagnose_v4.py
- PATH? `gpt2_surprisal.csv` (line 28)
- PATH? `words_labeled.csv` (line 29)
- PATH? `dataset_filler_v2.csv` (line 30)
- PATH? `gpt2_surprisal_propagated.csv` (line 31)

### diagnose_v5_final.py
- PATH? `dataset_filler_v2.csv` (line 27)
- PATH? `gpt2_surprisal_propagated.csv` (line 28)

### fix_linetoks.py
- READ `data/processed/words_labeled.csv` (line 3)

### generate_comparison_tables.py
- WRITE `table_accuracy_comparison.csv` (line 62)
- WRITE `table_coefficient_comparison.csv` (line 126)
- WRITE `table_gap_analysis.csv` (line 146)
- WRITE `table_pipeline_comparison.csv` (line 199)
- PATH? `figure_complete_comparison.png` (line 295)
- PATH? `table_accuracy_comparison.csv` (line 304)
- PATH? `table_coefficient_comparison.csv` (line 305)
- PATH? `table_gap_analysis.csv` (line 306)
- PATH? `table_pipeline_comparison.csv` (line 307)
- PATH? `figure_complete_comparison.png` (line 308)

### import_modelblocks.py
- PATH? `swbd_word_index.csv` (line 22)
- PATH? `swbd_5000_surprisal.csv` (line 23)
- PATH? `syntactic_surprisal_modelblocks_final.csv` (line 24)

### llm_surprisal_pilot_v3.py
- PATH? `dataset_reparandum.csv` (line 62)
- PATH? `clean_word_index.csv` (line 63)
- PATH? `llm_surprisal_pilot_v3.csv` (line 64)
- PATH? `llm_surprisal_pilot_v3_log.txt` (line 65)

### map_pipeline_io.py
- PATH? `pipeline_io_map.csv` (line 114)

### recompute_all_features.py
- PATH? `words_labeled.csv` (line 39)
- PATH? `lexical_surprisal.csv` (line 40)
- PATH? `dlt_costs.csv` (line 41)
- PATH? `pipeline_log.txt` (line 42)
- PATH? `.xml` (line 45)
- PATH? `.txt` (line 70)

### repo_inventory.py
- PATH? `$recycle.bin` (line 47)
- PATH? `.xml` (line 67)
- READ+WRITE? `repo_inventory.txt` (line 76)

### test_gpt2_surprisal.py
- no data file paths found

### write_glmm.py
- PATH? `dataset_filler.csv` (line 17)
- PATH? `dataset_reparandum.csv` (line 18)
- WRITE `.csv` (line 111)
- PATH? `.txt` (line 117)
- WRITE `table2_random_effects.csv` (line 154)

## Which script feeds which


## Files read but never written by any script (raw inputs)

- `clean_word_index.csv` (read by check_sentences.py)
- `dataset_filler_gpt2.csv` (read by check_foll_imputation.py, check_row_sets.py)
- `dataset_filler_multillm.csv` (read by check_foll_imputation.py, check_row_sets.py)
- `dataset_filler_v2.csv` (read by check_filler_join.py, check_fixed_merge.py, check_na_pattern.py, check_na_rate.py)
- `surprisal_gpt2.csv` (read by check_filler_join.py, check_fixed_merge.py, check_foll_imputation.py, check_foll_semantics.py, check_reparandum_merge.py)
- `swbd_5000_surprisal.csv` (read by compare_pcfg_modelblocks.py)
- `syntactic_surprisal.csv` (read by compare_pcfg_modelblocks.py)
- `table_multi_llm_comparison.csv` (read by 20_plot_multi_llm_comparison.py)
- `words_labeled.csv` (read by check_sentences.py, fix_linetoks.py)

## Files written but never read (final outputs or dead ends)

- `.csv` (written by 06_run_glmm.R, write_glmm.py)
- `foll_imputation_check.txt` (written by check_foll_imputation.py)
- `foll_semantics_check.txt` (written by check_foll_semantics.py)
- `multi_llm_comparison.png` (written by 20_plot_multi_llm_comparison.py)
- `nsaved to foll_imputation_check.txt` (written by check_foll_imputation.py)
- `nsaved to foll_semantics_check.txt` (written by check_foll_semantics.py)
- `nsaved to reparandum_merge_check.txt` (written by check_reparandum_merge.py)
- `nsaved to row_sets_check.txt` (written by check_row_sets.py)
- `nsaved to word_column_check.txt` (written by check_word_column.py)
- `pcfg_vs_modelblocks_comparison.csv` (written by compare_pcfg_modelblocks.py)
- `table1_filler_v2.csv` (written by 06_run_glmm_filler_v2.R)
- `table2_filler_v2.csv` (written by 06_run_glmm_filler_v2_table2.R)
- `table2_random_effects.csv` (written by 06_run_glmm.R, write_glmm.py)
- `table_accuracy_comparison.csv` (written by generate_comparison_tables.py)
- `table_coefficient_comparison.csv` (written by generate_comparison_tables.py)
- `table_gap_analysis.csv` (written by generate_comparison_tables.py)
- `table_pipeline_comparison.csv` (written by generate_comparison_tables.py)
- `word_column_check.txt` (written by check_word_column.py)
