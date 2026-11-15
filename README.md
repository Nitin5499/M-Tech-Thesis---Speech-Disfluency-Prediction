# Speech Disfluency Prediction — Replication Study

Replication of:
> Dammalapati et al. (2021). *Effects of Duration, Locality, and Surprisal in Speech Disfluency Prediction in English Spontaneous Speech.* SCiL 2021.

## Project Structure

```
speech-disfluency-prediction/
├── data/
│   ├── processed/          # generated CSVs (gitignored)
│   └── oanc/               # OANC corpus (gitignored - licensed)
├── results/                # output tables and accuracy files
├── src/
│   ├── 01_parse_xml.py     # Parse Switchboard NXT corpus
│   ├── 02_lexical_surprisal.py   # Trigram model on OANC
│   ├── 03_syntactic_surprisal.py # PCFG via ModelBlocks
│   ├── 04_dlt_costs.py     # DLT via Stanford Parser
│   ├── 05_build_dataset.py # Build balanced binary datasets
│   └── 06_run_glmm.R       # GLMM modeling in R (lme4)
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Install R 4.x and run:
```r
install.packages(c("lme4", "lmerTest"))
```

## Data Requirements

### Switchboard NXT Corpus
- Licensed from LDC: https://catalog.ldc.upenn.edu/
- Place extracted XML at: `data/switchboard-nxt-release-1.0/xml/`

### Open American National Corpus (OANC)
- Free download: https://anc.org/data/oanc/download/
- Place at: `data/oanc/`

### Penn Treebank (for syntactic surprisal)
- Licensed from LDC: https://catalog.ldc.upenn.edu/LDC99T42
- Sections 2-21 needed for PCFG training

## Running the Pipeline

```bash
# Step 1: Parse corpus
python src/01_parse_xml.py

# Step 2: Lexical surprisal (OANC trigram)
python src/02_lexical_surprisal.py

# Step 3: Syntactic surprisal (PCFG - ModelBlocks)
python src/03_syntactic_surprisal.py

# Step 4: DLT costs (Stanford Parser)
python src/04_dlt_costs.py

# Step 5: Build dataset
python src/05_build_dataset.py

# Step 6: GLMM in R
Rscript src/06_run_glmm.R
```

## Results

Results are saved in `results/` and compared against paper targets:

| Task | Our Accuracy | Paper Accuracy |
|------|-------------|----------------|
| Filler vs Fluent | TBD | 81.21% |
| Reparandum vs Fluent | TBD | 71.11% |

## Citation

```bibtex
@inproceedings{dammalapati2021effects,
  title={Effects of Duration, Locality, and Surprisal in Speech Disfluency Prediction},
  author={Dammalapati, Samvit and Rajkumar, Rajakrishnan and Ranjan, Sidharth and Agarwal, Sumeet},
  booktitle={Proceedings of the Society for Computation in Linguistics (SCiL)},
  year={2021}
}
```

## License

Code: MIT License
Dataset: See individual dataset licenses (LDC, OANC)