CONLLU_FILE is provided as ud.conllu, a subset of the Universal Depedencies text 
corpus, with some problematic lines removed. To run on any web-downloaded conllu 
files (https://github.com/UniversalDependencies/UD_English-LinES/blob/master/en_lines-ud-test.conllu), 
use conllu2sd_update.py instead. This requires pip install conllu. 

Run: 
1. python3 conllu2sd.py CONLLU_FILE OR python3 conllu2sd_update.py CONLLU_FILE
2. python3 cluster.py SD_FILE 35 112 11 12 29

Outputs: 
1. initial model with model condition : production probability
2. converged model with model condition : production probability
3. best.heads class : head probability
4. best.frames class likelihood : frame