import numpy as np
import h5py
from neuron import h
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd


# basic imports
run_file = './run_model_cori.hoc'
# run_volts_path = '../run_volts_bbp/'
paramsCSV = './params/params_bbp_full.csv'
orig_params = h5py.File('./params/params_bbp_full.hdf5', 'r')['orig_full'][0]
scores_path = './scores/'
objectives_file = h5py.File('./objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = './stims/stims_full.hdf5'
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
# Number of timesteps for the output volt.
ntimestep = 10000
# Value of dt in miliseconds
dt = 0.02

def run_model(param_set, stim_name_list):
    h.load_file(run_file)
    volts_list = []
    for elem in stim_name_list:
        curr_stim = h5py.File(stims_path, 'r')[elem][:]
        total_params_num = len(param_set)
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()
        volts_list.append(out)
    return np.array(volts_list)

def



if __name__ == "__main__":
    # run model and get target voltages
    for i in range(len(orig_params)):
        target_voltages = run_model(orig_params, opt_stim_list)
        
    # save voltages for viz    
    with open("volt_collector.npy", 'wb') as vs:
        np.save(vs, target_voltages)
