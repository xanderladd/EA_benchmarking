import sys
import csv

new_e_pas = sys.argv[1]
param_line = []
with open("params/params.csv", 'r') as f:
    for line in f.readlines():
        param_line.append(line)
        
param_line = param_line[0].split(" ")
param_line[1] = str(round(float(new_e_pas) / 10, 3)) +  "E+01"
param_line = (" ").join(param_line)

with open("params/params.csv", 'w') as f:
    f.write(param_line)

with open("params/params_map.csv", 'w') as f:
    for _ in range(5):
        f.write(param_line)
        if _ < 4:
            f.write("\n")
    