import os
import scipy.io as sio
from csv import reader, writer
import numpy as np

def update_cm(cms, NSets, global_rank=None):
    FNP_csv = '../Data/BasicConstSegP.csv'

    with open(FNP_csv, newline='') as csvfile:
        data_reader = reader(csvfile, delimiter=',')
        lines =[line for line in data_reader]
    if os.path.isfile('../Data/cm_map.txt'):
        cm_map = np.genfromtxt('../Data/cm_map.txt')
    else:
        cm_map =  np.where('2.000000000' == np.array(lines[6]))[0]
        np.savetxt('../Data/cm_map.txt',  cm_map)
        
    cm_map = cm_map.astype(np.int)
    new_line_6 = np.array([])
#     if cms is not None:
#         for cm in cms:
#             dummy_line6 = np.array(lines[6])
#             replacements = [format(c, '.9f') for c in cm]
#             #print(replacements)
#             #print(cm)
#             np.put(dummy_line6,cm_map, replacements)
#             new_line_6 = np.append(new_line_6, dummy_line6, axis=0)
#        # print(len(new_line_6), "new line is len ", len(new_line_6), '   orig line is ', len(dummy_line6))
#       #  print('cms was: ', len(cms))
#         lines[6] = new_line_6
        # TODO: how can this be cleaner? we are adding something in here
    
    #lines.insert(6,[format(NSets, '.9f')])
    # cannot rewrite one line, must rewrite whole csv
    ap_fn = '../Data/AllParams{}.csv'.format(str(global_rank))
    with open(ap_fn, 'a+') as csvfile:
        data_writer = writer(csvfile)
        print("MAX CM: ", np.max(cms))
        #data_writer.writerow(lines[6])
        data_writer.writerow([cm[0] for cm in cms])

    


        
      