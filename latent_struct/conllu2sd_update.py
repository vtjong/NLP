import conllu
import sys

def conllu_to_stanford_dep(conllu_text):
    """
    Converts a CoNLL-U formatted text to Stanford Dependency format.

    Args:
    - conllu_text (str): The CoNLL-U formatted text.

    Returns:
    - str: The text in Stanford Dependency format.
    """
    # Parse the CoNLL-U text
    sentences = conllu.parse(conllu_text)

    result_lines = []

    for sentence in sentences:
        # Initialize a mapping of IDs to word forms
        word_forms = {}
        
        # Iterate through the tokens and collect word forms
        for token in sentence:
            word_forms[token["id"]] = token["form"]
        
        # Iterate through the tokens to generate Stanford Dependency format
        for token in sentence:
            head_id = token["head"]
            if head_id == "0":
                head_form = "ROOT"
            else:
                head_form = word_forms.get(head_id, "")
            
            dep_rel = token["deprel"]
            word_id = token["id"]
            word_form = word_forms.get(word_id, "")

            # Create the Stanford Dependency format line
            dep_line = f"{dep_rel}({head_form}-{head_id}, {word_form}-{word_id})"
            result_lines.append(dep_line)
        
        # Add an empty line to separate sentences
        result_lines.append("")

    # Join the lines to form the output text
    return "\n".join(result_lines)

def convert_conllu_to_stanford_dep(input_file):
    """
    Converts a CoNLL-U formatted file to Stanford Dependency format.

    Args:
    - input_file (str): The path to the input CoNLL-U formatted file.
    - output_file (str): The path to the output file where the Stanford Dependency
      format data will be written.
    """
    output_file = input_file.replace(".conllu", ".sd")
    
    with open(input_file, "r") as ud_file, open(output_file, "w") as sd_file:
        ud_text = ud_file.read()
        stanford_dep_text = conllu_to_stanford_dep(ud_text)
        sd_file.write(stanford_dep_text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 conllu2sd.py CONLLU_FILE")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    convert_conllu_to_stanford_dep(input_file)