# Universal Dependencies to Stanford Dependency Converter

This tool is designed to convert Universal Dependencies (CONLLU) files into the Stanford Dependency format.

## Usage

To run this converter, follow the steps below:

1. **Prepare Your CONLLU File**:
   - Ensure you have your Universal Dependencies file (CONLLU_FILE) ready. A sample subset of the Universal Dependencies corpus is provided as `ud.conllu`, with some problematic lines removed.
   - If you want to run the converter on other Universal Dependencies files downloaded from the web, consider using `conllu2sd_update.py` instead. This script requires the `conllu` package, which can be installed using `pip install conllu`.
   - For additional Universal Dependencies files, visit the Universal Dependencies GitHub repository
   (https://github.com/UniversalDependencies).

2. **Conversion**:
   - Run the converter using one of the following commands:

   ```bash
   # Using conllu2sd.py
   python3 conllu2sd.py CONLLU_FILE
   # Using conllu2sd_update.py (for other downloaded files)
   python3 conllu2sd_update.py CONLLU_FILE

3. **Identify Latent Semantic Frames**:
   - After converting, identify latent semantic frames on the generated 
   Stanford Dependency file using the following command in bash:

   ```bash
   python3 cluster.py SD_FILE 35 112 11 12 29

4. **Outputs**:
    - initial model with model condition : production probability
    - converged model with model condition : production probability
    - best.heads class : head probability
    - best.frames class likelihood : frame

