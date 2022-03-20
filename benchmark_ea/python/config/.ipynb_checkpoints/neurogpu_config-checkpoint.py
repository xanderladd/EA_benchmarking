import csv
import pandas as pd
import os
import h5py
from mpi4py import MPI


try:
    nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))
except:
    nGpus = 1
    print('no GPUs found... exiting')
    
comm = MPI.COMM_WORLD
size = comm.Get_size()
total_rank = comm.Get_rank()
global_rank = comm.Get_rank() // nGpus
local_rank = comm.Get_rank() % nGpus
total_size = comm.Get_size()
global_size = size // nGpus
