# SimpleNLP

SimpleNLP is a lightweight NLP library that utilizes fundamental Python modules like `re`, `sys`, and `numpy`. This library includes a collection of scripts designed to handle various NLP tasks, including:

- **Part-of-Speech Tagging**: Hidden Markov Model (HMM) for Part-of-Speech tagging using the Viterbi algorithm, along with the Baum-Welsh algorithm for learning model parameters.

- **Syntactic Parsing**: Bottom-up CKY parsing for syntactic analysis.

- **Latent Semantic Frame Identification**: Tools for identifying latent semantic frames within text data.

## Project Structure

The library is organized into the following directories:

- `/cky`: Contains code for implementing syntactic bottom-up CKY parsing.
  
- `/latent_struct`: Contains code for latent semantic frame identification. For detailed information, please refer to the [README](latent_struct/README.md) in this directory.
  
- `/pos_tagger`: Contains code for Part-of-Speech tagging using HMMs.

## Note

While all code in this repository is functional, please note that the `/latent_struct` directory is the most actively maintained and developed. 
