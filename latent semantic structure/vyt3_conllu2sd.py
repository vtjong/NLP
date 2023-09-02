import sys

def ud_sd(sd_file, ud):
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

output_file = "ud.sd"
udfile_sdfile(sys.argv[1], output_file)