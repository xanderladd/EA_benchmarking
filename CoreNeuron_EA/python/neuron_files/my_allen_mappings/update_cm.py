import os
import scipy.io as sio
from csv import reader, writer
import numpy as np

def update_cm(cm):
    FNP_csv = '../Data/BasicConstSegP.csv'
    with open(FNP_csv, newline='') as csvfile:
        data_reader = reader(csvfile, delimiter=',')
        lines = np.array([line for line in data_reader])
    if os.path.isfile('../Data/cm_map.txt'):
        cm_map = np.genfromtxt('../Data/cm_map.txt')
    else:
        cm_map =  np.where('2.000000000' == np.array(lines[6]))[0]
        np.savetxt('../Data/cm_map.txt',  cm_map)
        
    cm_map = cm_map.astype(np.int)
    line6 = np.array(lines[6])
    np.put(line6,cm_map, np.array(cm).astype(str))
    lines[6] = line6
    # cannot rewrite one line, must rewrite whole csv
    with open(FNP_csv, 'w') as csvfile:
        data_writer = writer(csvfile)
        data_writer.writerows(lines)

        
      