# Part-of-Speech Tagger
This module contains an implementation of a POS tagger and sequence decoder 
for Hidden Markov Models (HMMs).

## `mle.py`

**Description:** This script trains an HMM using Maximum Likelihood Estimation (MLE) from annotated data.

- **Input:** Annotated training file in the format: `word state_count;word state_count;...`
- **Smoothing:** Applies add-1 smoothing to the transition matrix and adds a single emission count of `<unk>` to every latent state.
- **Training:** Trains model using MLE.
- **Output:** Saves the trained HMM to `model.hmm`

```bash
python3 mle.py hotcold.super
```

## `likelihood.py`

**Description:** This script computes the likelihood of an observation sequence given an HMM.

- **Input:** HMM file (`model.hmm`) and test sequence (`test.seqs`)
- **Forward Algorithm:** Computes the likelihood of the sequence using the forward algorithm.
- **Backward Algorithm:** Computes the likelihood of the sequence using the backward algorithm.
- **Output:** Saves forward and backward log likelihood and matrices to `hmm.likelihood`

```bash
python likelihood.py model.hmm test.seqs
```

## `decoding.py`

**Description:** This script determines the best hidden state sequence for an observation sequence using the Viterbi algorithm.

- **Input:** HMM file (`model.hmm`) and test sequence (`test.seqs`)
- **Viterbi Algorithm:** Determines the best hidden state sequence.
- **Output:** Saves the best hidden state sequence and log likelihood to `hmm.decoding`

```bash
python decoding.py model.hmm test.seqs
```

## `learning.py`

**Description:** This script tackles the Learning Problem. Given an observation sequence and a set of possible hidden states, it learns the HMM parameters. For simplicity, the script takes in an observation sequence and an HMM initialized with parameters, which are then updated based on the corpus.

- **Input:** HMM file (`model.hmm`) and a set of observation sequences (e.g., `hotcold.unsuper`).
- **HMM Initialization:** The HMM parameters are initialized as specified.
- **Learning Algorithm:** Utilizes the Forward-Backward algorithm to infer the correct parameters for the given HMM based on the provided observations.
- **Convergence:** Treats the model as converged when the forward sequence probability fails to increase by more than 0.1 log likelihood over 3 iterations.
- **Output:** Saves the updated HMM to `hmm.learning`.

```bash
python learning.py model.hmm hotcold.unsuper
```

