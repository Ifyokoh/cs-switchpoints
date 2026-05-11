# Code-Switching Switchpoints Analysis

Syntactic analysis of code-switching points across three language-pair treebanks
(Hindi–English, Telugu–English, Turkish–English).

The Research Question for this project is: Do bilingual speakers code-switch more often at syntactic boundaries than inside syntactic constituents? We operationalize syntactic position using dependency structure: a token pair is classified as *within-constituent* if the two tokens are directly connected in the dependency tree (one depends on the other, or both share the same head), and *boundary* otherwise. We then test whether boundary position predicts a higher probability of switching using a mixed-effects logistic regression model (`switch ~ relation + (1 | sent_id)`), with a random intercept per sentence to account for within-sentence clustering of pairs.

---

## Project structure

```
cs-switchpoints/
├── data/
│   ├── raw/                      # CoNLL-U treebanks
│   ├── clean/                    # cleaned token TSVs from prep_data.py
│   └── switchpairs/              # pair-level output from analyze_switchpoints.py
├── scripts/
│   ├── prep_data.py              # parse and clean raw treebanks
│   ├── analyze_switchpoints.py   # identify and classify code-switch points
│   ├── eda.py                    # exploratory data analysis and visualisations
│   └── mixedeffect.R             # mixed-effects logistic regression
│    
├── outputs/                      # plots and analysis results
├── print.txt                     # console output from prep_data.py and analyze_switchpoints.py
└── README.md
```


## Datasets

Three language-pair treebanks in CoNLL-U format, placed in `data/raw/`:

| Files | Language pair |
|---|---|
| [`hi_en.conllu`](https://github.com/UniversalDependencies/UD_Hindi_English-HIENCS/blob/master/qhe_hiencs-ud-train.conllu) | Hindi–English |
| [`te_en_.conllu`](https://github.com/UniversalDependencies/UD_Telugu_English-TECT) | Telugu–English |
| [`tr_en.conllu`](https://github.com/UniversalDependencies/UD_Turkish_English-BUTR/blob/master/qti_butr-ud-test.conllu) | Turkish–English |

To add a new language pair, drop a `.conllu` file into `data/raw/`.


## Requirements

**Python** (steps 1–3):
```
pip install -r requirements.txt
```
Required packages: `pandas`, `matplotlib`, `seaborn`.

**R** (step 4): install the following packages once inside R before running `mixedeffect.R`:
```r
install.packages(c("lme4", "ggplot2", "broom.mixed"))
```

---

## Usage

### 1. Preprocess treebanks
```
python scripts/prep_data.py
```

For each dataset, reads the raw CoNLL-U files and:

- Extracts five fields per token: ID, UPOS, HEAD, DEPREL, and language tag (from MISC).
- Drops tokens that cannot be code-switch points
- Writes `data/clean/{prefix}_tokens.tsv`.

### 2. Identify and classify switch points
```
python scripts/analyze_switchpoints.py
```

Reads the cleaned `_tokens.tsv` files and for each dataset:

- Groups tokens back into sentences and walks every adjacent pair.
- Labels each pair with two properties:
  - **switch** — `true` if the two tokens are in different languages, `false` if same,
    `unknown` if either language tag is missing.
  - **relation** — `within` if the pair is syntactically connected (direct dependency
    in either direction, or shared head word); `boundary` otherwise; `unknown` if
    either HEAD is missing.
- Writes `data/switchpairs/{prefix}_pairs.tsv`.

### 3. Exploratory data analysis
```
python scripts/eda.py
```

Reads `data/switchpairs/*_pairs.tsv` and writes the following to `outputs/`:

| Output | Description |
|---|---|
| `sentence_length_distribution.svg` | Histogram of sentence length (pairs/sentence) per dataset |
| `within_boundary_by_switch.svg` | Within/boundary split for switch vs same-language pairs |
| `pos_top10.svg` | Top-10 POS pairs at switch points per dataset |
| `deprel_by_relation.svg` | Dependency role breakdown at switch points (within vs boundary) |
| `eda_summary.txt` | Numerical summary of all sections |

### 4. Mixed-effects logistic regression
```
Rscript scripts/mixedeffect.R
```

Fits `Switch ~ Boundary + (1 | Sentence)` separately for each language pair and writes to `outputs/`:

| Output | Description |
|---|---|
| `glmm_odds_ratio_plot.svg` | Forest plot of odds ratios with 95% CIs |
| `glmm_predicted_prob_plot.svg` | Predicted P(switch) for within vs boundary (Hindi–English) |
| `stats_summary.txt` | Full numerical results table |

