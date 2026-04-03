# Code-Switching Switchpoints Analysis

Syntactic analysis of code-switching points across three language-pair treebanks
(Hindi-English, Telugu-English, Turkish-English).



## Project structure

```
cs-switchpoints/
├── data/
│   ├── raw/                      # CoNLL-U treebanks 
│   ├── clean/                    # cleaned token TSVs from prep_data.py 
│   └── switchpairs/              # pair-level output from analyze_switchpoints.py 
├── scripts/
│   ├── prep_data.py              # parse and clean raw treebanks
│   └── analyze_switchpoints.py   # identify and classify code-switch points
├── print.txt                     # shows the results from the scripts reports
└── README.md
```


## Datasets

Three language-pair treebanks in CoNLL-U format, placed in `data/raw/`:

| Files | Language pair |
|---|---|
| [`hi_en.conllu`](https://github.com/UniversalDependencies/UD_Hindi_English-HIENCS/blob/master/qhe_hiencs-ud-train.conllu) | Hindi-English |
| [`te_en_.conllu`](https://github.com/UniversalDependencies/UD_Telugu_English-TECT)| Telugu-English |
| [`tr_en.conllu`](https://github.com/UniversalDependencies/UD_Turkish_English-BUTR/blob/master/qti_butr-ud-test.conllu) | Turkish-English |

To add a new language pair
requires only dropping a `.conllu` file into `data/raw/`


## Usage
1. Run the preprocessing script to parse and clean treebanks
```
python scripts/prep_data.py
```

The script does the following:

For each dataset, reads the raw CoNLL-U files and:

- Extracts five fields per token: ID, UPOS, HEAD, DEPREL, and language tag (from MISC).
- Drops tokens that cannot be code-switch points:
  - `UPOS=PUNCT` — punctuation has no language identity
  - `Lang=univ` — explicitly language-neutral tokens
  - `Lang=ne` — named entities
  - `Lang=acro` — acronyms
- Reports data-quality diagnostics: missing HEAD values, missing language tags,
  and monolingual sentences broken down by language.


2. Run the analysis script to Identify and classify switch points
```
python scripts/analyze_switchpoints.py
```

The script reads the cleaned `_tokens.tsv` files and for each dataset:

- Groups tokens back into sentences and walks every adjacent pair.
- Labels each pair with two properties:
  - **switch** — `true` if the two tokens are in different languages, `false` if same,
    `unknown` if either language tag is missing.
  - **relation** — `within` if the pair is syntactically connected (direct dependency
    in either direction, or shared head word); `boundary` otherwise; `unknown` if
    either HEAD is missing.
- Reports a per-dataset summary: total pairs, switch rate, and within/boundary
  breakdown for both switch and same-language pairs.


