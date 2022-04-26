import csv
import os
import h5py
# import utils


#nCpus =  multiprocessing.cpu_count()

run_volts_path ='../../allen_data/' 


target_volts_path = run_volts_path+'target_volts_485835016.hdf5'
stims_path = run_volts_path+'stims_485835016.hdf5'
paramsCSV = run_volts_path+'params_bbp_full_gpu_tuned_10_based.csv'
objectives_path = run_volts_path+'allen485835016_objectives.hdf5'
allen_param_path = run_volts_path+'params_bbp_full_allen_gpu_tune.hdf5'

target_volts_hdf5 = h5py.File(target_volts_path, 'r')
stim_file = h5py.File(stims_path, 'r')
objectives_file = h5py.File(objectives_path, 'r')
orig_params = h5py.File(allen_param_path, 'r')['orig_full'][0]

scores_path = '../../scores/'
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list']


params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
vs_fn = '../Data/VHotP'
# Number of timesteps for the output volt.
ntimestep = 10000
stim_names = list([e for e in opt_stim_name_list if len(e) < 7])



custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

try:
    nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))
except:
    nGpus = 1
    
    

    
## xander custom stim / weight list
import numpy as np



good_sfs = [b'AHP_depth', b'Spikecount', b'chi_square_normal', b'AP_height',  b'AP_duration', b'ISI_values', b'AP_width',b'fast_AHP',  b'AP_amplitude_from_voltagebase',b'AHP_time_from_peak',b'AP_fall_rate_change', b'AP_fall_time', b'AP_height',b'isi']
good_stims = ['18','55_2','75']

mask = np.isin(score_function_ordered_list,good_sfs)
weight_list = np.ones(shape=(79))
weight_list[mask] = 20
opt_weight_list = np.tile(weight_list, len(good_stims))
opt_stim_name_list = good_stims