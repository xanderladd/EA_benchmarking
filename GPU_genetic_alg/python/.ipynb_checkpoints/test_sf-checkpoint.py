import numpy as np
import math
import efel
#import matplotlib.pyplot as plt
import time as timer



f =h5py.File("../Data/0.hdf5","r")
import pdb; pdb.set_trace()
curr_data_volt =f["data_volt{}{}".format(i,j)][:]
curr_target_volt =f["target_volt{}{}".format(i,j)][:]