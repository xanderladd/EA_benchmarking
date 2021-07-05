import csv
import pandas as pd
import os
import numpy as np
import h5py
import nrnUtils
import multiprocessing
from mpi4py import MPI
import utils


nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
#nCpus =  multiprocessing.cpu_count()
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()


# print("USING nGPUS: ", nGpus, " and USING nCPUS: ", nCpus)
# print("Rank: ", global_rank)
# CPU_name = MPI.Get_processor_name()
# print("CPU name", CPU_name)


visualize=True
# original model
# objectives_file = h5py.File('./objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
# target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
# stims_path = '../stims/allen_data_stims_10000.hdf5'
# scores_path = '../../scores/'
#params_opt_ind = np.arange(15)


# new model
objectives_file = h5py.File('./objectives/allen485835016_objectives.hdf5', 'r')
target_volts_path = './target_volts/target_volts_485835016.hdf5'
stims_path = '../stims/stims_485835016.hdf5'
scores_path = '/global/cscratch1/sd/zladd/allen_dataset/scores_14_params'
params_opt_ind = np.arange(15)
passive=False
visualize=False


# objectives_file = h5py.File('./objectives/allen485835016_objectives_passive.hdf5', 'r')
# target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'
# stims_path = '../stims/stims_485835016_passive.hdf5'
# scores_path = '../../scores/'
# passive=True
#params_opt_ind = [0, 1, 6, 9, 13, 14]

#params_opt_ind = [0,1,4,5, 13,14]
#params_opt_ind = [0,1,6,9,13,14]
#params_opt_ind = np.arange(15)




run_file = './run_model_cori.hoc'
paramsCSV = '../params/params_bbp_full_gpu_tuned_10_based.csv'
orig_params = np.array(nrnUtils.readParamsCSV(paramsCSV))[:,1].astype(np.float64)
# orig_params = [0.0000022060, -66.5600000000, 4.0383444556,3.1263250613,0.0047147854,2.8515288117, 5.1781678079, 0.1096452854,0.0005345695, 0.1578655731,0.0016590198, 3.7929209114, 0.0036489355,0.0002609501, orig_params[14]]
if 'opt_weight_list' in objectives_file.keys():
    opt_weight_list = objectives_file['opt_weight_list'][:]
else:
    opt_weight_list = np.ones(60000)
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list']
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

print(target_volts_hdf5.keys(), opt_stim_name_list)
ap_tune_stim_name = '18'
ap_tune_weight = 0
#params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
data_dir = '../Data/allenData/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'
stim_file = h5py.File(stims_path, 'r')
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

templateCSV = "../params/params_bbp_full_gpu_tuned_10_based.csv"

# Number of timesteps for the output volt.
ntimestep = 10000

stim_names = list([e.decode('ascii') for e in opt_stim_name_list if len(e.decode('ascii')) < 7])

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']


