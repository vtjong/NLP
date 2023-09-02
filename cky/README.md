# Probabilistic Context-Free Grammar (PCFG) Toolkit
This toolkit provides a set of Python scripts for working with Probabilistic 
Context-Free Grammars (PCFGs) in Natural Language Processing (NLP). It includes 
scripts for PCFG estimation using Maximum Likelihood Estimation (MLE), CKY parsing 
with Viterbi decoding, and parser evaluation against gold standard parses.

## `mle.py`

**Description:** This script estimates a Probabilistic Context-Free Grammar (PCFG) 
using Maximum Likelihood Estimation (MLE) from syntactic annotations.

- **Input:** Annotated training file and a float value.
- **Output:** The script saves the estimated PCFG to `model.pcfg`.

```bash
python mle.py training_file 0.8
```

## `cky.py`

**Description:** This script generates the best CKY parses for sequences of word 
tokens using Viterbi decoding.

- **Input:** PCFG file (`model.pcfg`) and an unannotated test file.
- **Output:** Saves forward and backward log likelihood and matrices to `hmm.likelihood`

```bash
python cky.py model.pcfg test_file

```

## `eval.py`

**Description:** This script evaluates the quality of output parses against gold standard parses.

- **Input:** Parser output file (`output.parses`) and a file with gold parses (`gold.parses`).
- **Output:** The script calculates and outputs unlabeled precision, unlabeled recall, and unlabeled F1 score for each gold parse.

```bash
python eval.py output.parses gold.parses
```