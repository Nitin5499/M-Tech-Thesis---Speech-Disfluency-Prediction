# Predicting Speech Disfluency

M.Tech. (Data Engineering) project, MTP-1, IIT Jodhpur
Nitin Kumar (M25DE2047) · Supervisor: Dr. Sidharth Ranjan

This repository holds the full pipeline, the model-ready datasets, the result tables and a small toolkit.
The project predicts two kinds of disfluency in the Switchboard NXT corpus, using word duration, surprisal and dependency-locality costs, and prosody.

- **Filler task:** at a position after a clean word, is there a filler (uh, um and similar)?
- **Reparandum task:** is a clean word the start of a stretch the speaker goes on to repair?

Each task is a binary classification fitted with a logistic mixed-effects model (random intercept per word), scored with 5-fold cross-validation (seed 42, threshold 0.5) and a likelihood-ratio test.

## Main results

| Experiment | Filler accuracy | Reparandum accuracy |
|---|---|---|
| Always guess the majority class | 59.59 % | 52.63 % |
| Baseline (duration, trigram + syntactic surprisal, IC, SC) | 78.56 % | 73.85 % |
| + language-model surprisal (5 models) | +1.03 to +1.45 points | −0.03 to +0.15 points |
| + prosody (break index, accent) | see `results/` | see `results/` |
| Toolkit, text only, 8 features, with word effects | 72.85 % | 70.71 % |
| Toolkit, text only, 8 features, fixed effects only | 67.60 % | 56.12 % |

Exact tables, coefficients and AUC values are in `results/`.

## Repository layout

```
src/                 pipeline scripts (Python and R), numbered in run order
toolkit/             command-line tool and Streamlit page
  build_ngram.py     builds the trigram counts from OANC
  disfluency_toolkit.py
  App.py
  models/            exported coefficients, word effects, metadata
data/processed/      model-ready datasets (see below)
results/             result tables and figures (csv, txt, png)
pipeline_io_map.md   which script reads and writes which file
```

## Data

The corpora are licensed or large and are **not** in the repository. Place them as follows before running.

| Corpus | Used for | Folder |
|---|---|---|
| Switchboard NXT (words, disfluency, accent, break annotation) | all tasks | `data/switchboard-nxt/` |
| Open American National Corpus (OANC) | trigram surprisal | `data/oanc/` |
| Penn Treebank WSJ 02–21 + ModelBlocks release | syntactic surprisal | `data/penn-treebank/`, `data/modelblocks/` |

Included in `data/processed/`:

- `dataset_filler_fixfoll_multillm.csv`: 37,543 rows, filler task, with surprisal of five language models.
- `dataset_reparandum_multillm.csv`: 40,732 rows, reparandum task, with surprisal of five language models.

With these two files the models of Experiments II and IV can be re-fitted without the raw corpora.

## Requirements

- Python 3.13 with: `pandas`, `numpy`, `torch`, `transformers`, `spacy`, `streamlit`, `matplotlib`, `lxml`
- spaCy model: `python -m spacy download en_core_web_sm`
- R 4.5 with `lme4`, `pROC`, `dplyr`, `readr`
- ModelBlocks needs a Linux shell (WSL on Windows) for the syntactic surprisal step

```
python -m venv venv
venv\Scripts\activate
pip install pandas numpy torch transformers spacy streamlit matplotlib lxml
python -m spacy download en_core_web_sm
```

## Run order

Run every command from the project root. Each script reads the outputs of the earlier ones.

| Stage | Scripts | Output |
|---|---|---|
| 1. Parse corpus | `src/01_parse_xml.py` | `words_labeled.csv` |
| 2. Clean corpus | `src/03a_clean_corpus.py` | `clean_word_index.csv`, `clean_sentences.txt` |
| 3. Lexical surprisal | `src/02_lexical_surprisal.py` | `lexical_surprisal.csv` |
| 4. Dependency costs | `src/04_dlt_costs.py` | `dlt_costs.csv` |
| 5. Syntactic surprisal | `src/03b_prepare_modelblocks.py`, ModelBlocks run (`src/03b_parallel_modelblocks.py`), `src/03_syntactic_surprisal_modelblocks.py`, `src/04b_merge_modelblocks.py` | `syntactic_surprisal.csv` |
| 6. Baseline datasets | `src/05_build_dataset.py`, `src/05_build_dataset_v2.py` | `words_features.csv`, `dataset_filler.csv`, `dataset_filler_v2.csv`, `dataset_reparandum.csv` |
| 7. Baseline models | `src/06_run_glmm.R`, `src/06_run_glmm_filler_v2.R`, `src/06_run_glmm_filler_v2_table2.R` | Tables of Experiment I |
| 8. GPT-2 surprisal | `src/14_gpt2_surprisal.py`, `src/15_merge_gpt2_dataset.py`, `src/16_run_glmm_gpt2.R`, `src/16_gpt2_vs_modelblocks_correlation.py` | `gpt2_surprisal.csv`, `table_gpt2_comparison.csv` |
| 9. Five language models | `src/17_multi_llm_surprisal.py`, `src/18_merge_multi_llm_FIXED_v2.py`, `src/18b_fix_filler_foll_offset.py` | `surprisal_<model>.csv`, `dataset_*_multillm.csv`, `dataset_filler_fixfoll_multillm.csv` (filler dataset with the corrected following-word features) |
| 10. Model comparison | `src/19_run_glmm_multi_llm_FIXED_v2.R`, `src/19b_run_glmm_filler_fixfoll.R`, `src/19c_run_glmm_multi_llm_coef_auc.R` | `table_multi_llm_comparison.csv` (Experiment II) |
| 11. Plot | `src/20_plot_multi_llm_comparison.py` | Figure 5.1 |
| 12. Prosody parsing | `src/07_parse_phonwords.py`, `src/09_parse_accents.py`, `src/10_parse_breaks.py` | phonword, accent and break indexes |
| 13. Bridge and features | `src/08_bridge_terminals_phonwords.py`, `src/11_build_prosodic_features.py` | `word_prosody_bridge.csv`, `prosodic_features.csv` |
| 14. Prosody datasets | `src/12_merge_prosodic_dataset.py`, `src/12b_rebuild_filler_prosodic_fixfoll.py` | four prosody datasets |
| 15. Prosody models | `src/13_run_glmm_prosodic.R`, `src/13b_run_glmm_prosodic_auc.R`, `src/13c_run_glmm_prosodic_coef.R`, `src/13d_run_glmm_prosody_vs_gpt2.R` | Tables of Experiment III |
| 16. Syllables and sources | `src/20_syllable_analysis.py`, `src/21_prosody_by_source.py` | accent by syllables, by source |
| 17. Sign check | `src/22_check_lex_sign.R` | `check_lex_sign.csv` |
| 18. Toolkit models | `toolkit/build_ngram.py`, `src/24_export_toolkit_models.R` | `toolkit/models/` |

Python scripts: `python src/<script>.py`. R scripts: `Rscript src/<script>.R`.

### Quick path (from the two included datasets)

```
python toolkit/build_ngram.py
Rscript src/24_export_toolkit_models.R
Rscript src/19_run_glmm_multi_llm_FIXED_v2.R
```

## Toolkit

Type a sentence, get the probability of a filler and of a reparandum at every gap between two words.

```
python toolkit/disfluency_toolkit.py "so i went to the store and then we left"
python toolkit/disfluency_toolkit.py --check
streamlit run toolkit/App.py
```

- A gap sits between word *i* and word *i+1*. The previous predictors belong to word *i*, the following ones to word *i+1*.
- Eight text-only predictors: trigram surprisal, GPT-2 surprisal, integration cost and storage cost, each for the previous and following word.
- Probability = σ(β₀ + Σ βₖ zₖ + uᵥ), where zₖ is the standardised predictor and uᵥ is the learned random intercept of the word before the gap (0 for a word not in training).
- The probabilities are relative to a training sample with 40.4 % fillers and 47.4 % reparanda, so read them as a ranking of gaps.
- `--check` recomputes the features of the pipeline and compares them with the toolkit.

## Notes

- Random seed 42 everywhere; 5 folds; threshold 0.5.
- Trigram model: add-one smoothing, vocabulary 506,434 types; an unseen context gives log₂ V = 18.95 bits.
- Language models: GPT-2 (124M), DistilGPT-2 (82M), GPT-2-medium (355M), Pythia-160m, OPT-125m.