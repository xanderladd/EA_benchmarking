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
import csv
import bluepyopt as bpop
import shutil, errno
import pandas as pd
import nrnUtils
import utils
from matplotlib.backends.backend_pdf import PdfPages

paramsCSV = '../params/params_bbp_full_gpu_tuned_10_based.csv'
vs_fn = '/tmp/Data/VHotP'

param_tbl = np.array(nrnUtils.readParamsCSV(paramsCSV))

orig_params = param_tbl[:,1].astype(np.float64)


objectives_file = h5py.File('./objectives/allen485835016_objectives_passive.hdf5', 'r')
target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]

opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list if len(e.decode('ascii')) < 7]

target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
targetFile = [target_volts_hdf5[s][:] for s in opt_stim_list]

cm1 = np.genfromtxt("cm1.csv", delimiter=",")
cm5 = np.genfromtxt("targetVolts_cm.5.csv", delimiter=",")
cm20 = np.genfromtxt("targetVolts_cm20.csv", delimiter=",")

def run_model(stim_ind):
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
    volts_fn =  vs_fn + str(stim_ind) + '_0.dat'
    if os.path.exists(volts_fn):
        #print("removing ", volts_fn, " from ", global_rank)
        os.remove(volts_fn)
    #!{'../bin/neuroGPU'+str(global_rank),str(stim_ind), str(global_rank)}
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,  # <-- redirect stderr to stdout
                    bufsize=1)
    
    with p_object.stdout:
        for line in iter(p_object.stdout.readline, b''):
            print(line)
    p_object.wait()
    print(p_object.stderr)

    return p_object

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)



param_values =  np.array(orig_params).reshape(1,-1)
num_params = len(orig_params)
param_values = np.repeat(param_values, 5, axis=0)
param_values[1,14] = .5
param_values[2,14] = 1
param_values[3,14] = 20
cms = param_values[:,-1:]
param_values = param_values[:,:-1]
allparams_from_mapping(param_values,cms=cms)


for idx, (cmone,cmonefive,cmtwenty,targ) in enumerate(zip(cm1,cm5,cm20,targetFile)):
    run_model(idx)
    all_volts = nrnMread("/tmp/Data/VHotP{}_0.dat".format(idx))
    orig_volts = all_volts[:10000]
    point5 = all_volts[10000:20000]
    one = all_volts[20000:30000]
    twenty = all_volts[30000:40000]
    plt.figure()
    plt.plot(point5, color="orange", label="neuroGPU .5 cm")
    plt.plot(one, color="black", label="neuroGPU 1 cm")
    plt.plot(twenty, color="pink", label="neuroGPU 20 cm")

    plt.plot(cmonefive, color="blue", label=".5 cm")
    plt.plot(cmone, color="red", label="1 cm")
    plt.plot(targ, color='gold', label='target')
    plt.plot(cmtwenty, color='green', label="20 cm")
    plt.legend()
    plt.savefig("neuron_v_neurogpu")
    print(1/0)

