import sys

def ud_sd(sd_file, ud):
    """
    Parameters:
    - sd_file (file): The file object to which the Stanford Dependency 
    format data will be written.
    - ud (dict): A dictionary containing Universal Dependencies data. 

    Each entry should have the following format:
      {
          'id': (head, dependencies),
          ...
      }
      where 'id' is the word ID, 'head' is the head of the word, and 
      'dependencies' is a string of dependencies
      separated by '|'.
    """
    for id, v in ud.items():
        fr = v[0]
        deps =  v[1]

        deps = deps.split("|")
        for dep in deps:
            dep = dep.split(":")
            for index, value in enumerate(dep):
                if index == 0:
                    dep_id = value
                elif index == 1:
                    dep_tag = value
                else:
                    dep_tag = dep_tag + "_" + value

            if dep_id == "0":
                to_tag = "ROOT"
            else:
                to_val = ud[dep_id]
                to_tag = to_val[0]

            sd_line = dep_tag + "(" + to_tag + "-" + dep_id + ", " + fr + "-" + id + ")"
            sd_file.write(sd_line + "\n")
    sd_file.write("\n")


def udfile_sdfile(ud_file, sd_file):
    """
    Converts a Universal Dependencies file (CONLL-U) to a Stanford Dependency file.

    This function takes a Universal Dependencies file `ud_file`, 
    reads its content, and converts it into Stanford Dependency format, 
    which is then written to the specified `sd_file`.

    Parameters:
    - ud_file (str): The path to the Universal Dependencies file (CONLL-U) to be converted.
    - sd_file (str): The path to the output Stanford Dependency file where the 
    converted data will be written.

    Example Usage:
    ```
    udfile = 'input.conllu'
    sdfile = 'ud.sd'

    udfile_sdfile(udfile, sdfile)
    ```

    This example reads 'input.conllu' in CONLL-U format, 
    converts it to Stanford Dependency format, and saves the result in 'ud.sd' file.
    """
    with open(sd_file, "w") as sd_file:
        with open(ud_file, "r") as ud_file:
            lines = ud_file.read().strip().split("\n")
            connllu = {}
            for line in lines:
                if line != "":
                    line = line.split()
                    if line[0] == "#":
                        continue
                    connllu[line[0]] = (line[1], line[8])
                else:
                    ud_sd(sd_file, connllu)
                    connllu = {}
            if len(connllu) != 0:
                ud_sd(sd_file, connllu)

output_file = "en_ewt-ud-dev.sd"
udfile_sdfile(sys.argv[1], output_file)