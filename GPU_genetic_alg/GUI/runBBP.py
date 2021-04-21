import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from extractModel_mappings import   allparams_from_mapping
import subprocess



paramsCSV = '../params/params_bbp_full.csv'
#orig_params = np.array(np.array(nrnUtils.readParamsCSV(paramsCSV))[:,1], dtype=np.float64)
orig_params = h5py.File('../params/params_bbp_full.hdf5', 'r')['orig_full'][0]
print(orig_params.shape)
scores_path = '../scores/'
objectives_file = h5py.File('../objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = '../stims/stims_full.hdf5'
stim_file = h5py.File(stims_path, 'r')
#target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
#target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#params_opt_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
params_opt_ind = np.arange(24) 
model_dir = '..'
data_dir = model_dir+'/Data/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'
target_volts = np.genfromtxt("../Data/target_volts_BBP19.csv",delimiter = ',')

if not os.path.isdir('/tmp/Data'):
    os.mkdir('/tmp/Data')


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


import pandas as pd

def readParamsCSV(fileName):
    fields = ['Param name', 'Base value','Lower bound', 'Upper bound']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    
    paramsList = [tuple(x) for x in df.values]
    return paramsList


import bluepyopt as bpop

opt_ind = np.arange(24) 
data = readParamsCSV(paramsCSV)
data = np.array([data[i] for i in opt_ind])
pmin = np.array((data[:,2]), dtype=np.float64)
pmax = np.array((data[:,3]), dtype=np.float64)
# make this a function
fixed = {}
params = []
for param_idx in range(len(orig_params)):
    if np.isclose(orig_params[param_idx],pmin[param_idx],rtol=.001) and np.isclose(pmin[param_idx],pmax[param_idx],rtol=.001):
        print(orig_params[param_idx], " idx : ", param_idx)
        fixed[param_idx] = orig_params[param_idx]
    else:
        params.append(bpop.parameters.Parameter(orig_params[param_idx], bounds=(pmin[param_idx],pmax[param_idx])))
        
   # convert the allen data and save as csv
import csv
ntimestep = 100000
def convert_allen_data():
    """
    Function that sets up our new allen data every run. It reads and writes every stimi
    and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
    """
    for i in range(len(opt_stim_name_list)):
        old_stim = "../Data/Stim_raw{}.csv"
        old_time = "../Data/times_{}.csv"
        if os.path.exists(old_stim) :
            os.remove(old_stim)
            os.remove(old_time)
    for i in range(len(opt_stim_name_list)):
        stim = opt_stim_name_list[i].decode("utf-8")
        dt = .02 # refactor this later to be read or set to .02 if not configured
        f = open ("../Data/times{}.csv".format(i), 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        current_times = [dt for i in range(ntimestep)]
        wtr.writerow(current_times)
        f.close()
        np.savetxt("../Data/Stim_raw{}.csv".format(i), 
                   stim_file[stim][:],
                   delimiter=",")
def run_model(stim_ind,real_ind):
    """
    Parameters
    -------------------------------------------------------
    stim_ind: index to send as arg to neuroGPU 
    params: DEPRECATED remove

    Returns
    ---------------------------------------------------------
    p_object: process object that stops when neuroGPU done
    """
    global_rank = 0
    volts_fn = vs_fn + str(stim_ind) + '.h5'
    if os.path.exists(volts_fn):
        #print("removing ", volts_fn, " from ", global_rank)
        os.remove(volts_fn)
    cmd = 'ls -l'
    #!{'../bin/neuroGPU'+str(global_rank),str(stim_ind), str(global_rank)}
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,  # <-- redirect stderr to stdout
                    bufsize=1)
    print(p_object, stim_ind)
    with p_object.stdout:
        for line in iter(p_object.stdout.readline, b''):
            print(line),
    p_object.wait()
    print(p_object.stderr)
    os.rename(volts_fn,'/tmp/Data/VHotP'+str(real_ind)+'.h5')

    return p_object

def stim_swap(idx, i):
    """
    Stim swap takes 'idx' which is the stim index % 8 and 'i' which is the actual stim idx
    and then deletes the one at 'idx' and replaces it with the stim at i so that 
    neuroGPU reads stims like 13 as stim_raw5 (13 % 8)
    """
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')

def getVolts(idx):
    '''Helper function that gets volts from data and shapes them for a given stim index'''
    fn = vs_fn + str(idx) +  '.h5'    #'.h5' 
    curr_volts =  nrnMreadH5(fn)
    #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    #curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts




paramset = np.array(pickle.load( open( "../python/best_indvs_final_31_Oct_2020.pkl", "rb" ) ))
for reinsert_idx in fixed.keys():
    paramset = np.insert(np.array(paramset), reinsert_idx, fixed[reinsert_idx], axis = 1)

print(paramset[0])
print(orig_params)

nstims = len(opt_stim_list)
convert_allen_data()
#allparams_from_mapping(paramset)


###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################
param_values =  np.array(orig_params).reshape(1,-1)
param_values = np.repeat(param_values, 10, axis=0)
print(param_values.shape, "pvals shape!!!!!!!!")
allparams_from_mapping(param_values)
###### TEN COPIES OF ORIG PARAMS FOR DEBUG ################# 

for i in range(0,3):
    if i != 0:
        p_object = run_model(0,i)
        p_object.wait()
        #getVolts(0)
        stim_swap(0,i)

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

for file in os.listdir('../Data'):
    if 'h5' in file:
        print("removing: ", file)
        os.remove('../Data/' + file)

for file in os.listdir('/tmp/Data'):
    shutil.move("/tmp/Data/"+ file, "../Data/" + file)


    