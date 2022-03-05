import os
import re
import numpy as np

bad_extensions = [".png",".csv",".tex", ".ipynb", ".gitignore", '.txt']
output_folders = [file for file in os.listdir("python/outputs") if bad_extensions]


nodes = []
cores = []
offsprings = []
stims = []
stim = []
sfs = []

for file in os.listdir("python/summit_outputs"):
    
    skip = False
    for bad_ext in bad_extensions:
        if bad_ext in file:
            skip = True
            
    if skip: 
        continue
        
        
    numbers = re.findall(r'\d+', file)
    print("FILE :", file)
    node = numbers[0]
    core = numbers[1]
    offspring = numbers[2]
    stim = numbers[3]
    sf = numbers[4]
    try:
        version = numbers[5]
        print("SKipping versioned output")
        continue
    except:
        print("not a verioned output")
    if int(node) > 12:
        continue
    nodes.append(node)
    cores.append(core)
    offsprings.append(offspring)
    stims.append(stim)
    sfs.append(sf)

    print(node,core,offspring,stims)

lines = ['nGens','offspring', 'cpuTrials', 'N','n_stims','n_sfs']
fields = [np.repeat(1, len(offsprings)), np.array(offsprings), np.array(cores), np.array(nodes), np.array(stims), np.array(sfs)]
with open('aggregated_plan.txt', 'w') as f:
    for field,line in zip(fields, lines):
        curr_line = line + "=" + ",".join(field.astype(str)) + "\n"
        print(curr_line)
        f.write(curr_line)
    f.write('clean=False')
    
    
