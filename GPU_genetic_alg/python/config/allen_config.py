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


run_file = './run_model_cori.hoc'
run_volts_path ='../' # archival: '../../run_volts_bbp_full_gpu_tuned/'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
scores_path = '../../scores/'
objectives_file = h5py.File('./objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list']
stims_path = run_volts_path+'/stims/allen_data_stims_10000.hdf5'
target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
ap_tune_stim_name = '18'
ap_tune_weight = 0
params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
data_dir = '../Data/allenData/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'
stim_file = h5py.File(stims_path, 'r')
target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

templateCSV = "../params/params_bbp_full_gpu_tuned_10_based.csv"

# Number of timesteps for the output volt.
ntimestep = 10000

stim_names = list([e.decode('ascii') for e in opt_stim_name_list])

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']


# model = "bbp"
# peeling ="potassium"
# peeling = "potassium"
# date = "02_04_2021"
# objectives_file = h5py.File('../../objectives/multi_stim_without_sensitivity_bbp_' + peeling + "_" + date + '_stims.hdf5', 'r')
# opt_weight_list = objectives_file['opt_weight_list'][:]
# opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
# score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
