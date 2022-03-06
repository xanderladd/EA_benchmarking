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
#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")



run_volts_path = '../'
# paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
# objectives_file = h5py.File('../objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
# orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
# stims_path = run_volts_path+'/stims/allen_data_stims_10000.hdf5'
# target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'

paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based_allzero.csv'
objectives_file = h5py.File('../objectives/allen485835016_objectives_passive.hdf5', 'r')
#orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
orig_params = np.array(nrnUtils.readParamsCSV(paramsCSV))[:,1]
stims_path = run_volts_path+'stims/stims_485835016_passive.hdf5'
target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'



scores_path = './scores/allen_scores/'
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
target_volts_hdf5 = [target_volts_hdf5[s][:] for s in opt_stim_name_list]

ap_tune_stim_name = '18'
stim_file = h5py.File(stims_path, 'r')
ntimestep = 10000
vs_fn = '../Data/VHotP'


nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])



if not os.path.isdir('/tmp/Data'):
    os.mkdir('/tmp/Data')
    


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def readParamsCSV(fileName):
    fields = ['Param name', 'Base value','Lower bound', 'Upper bound']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    
    paramsList = [tuple(x) for x in df.values]
    return paramsList


# def prepare_data():
#     """
#     Function that sets up our new allen data every run. It reads and writes every stimi
#     and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
#     """
#     for i in range(len(opt_stim_name_list)):
#         old_stim = "../Data/Stim_raw{}.csv"
#         old_time = "../Data/times_{}.csv"
#         if os.path.exists(old_stim) :
#             os.remove(old_stim)
#             os.remove(old_time)
#     for i in range(len(opt_stim_name_list)):
#         stim = opt_stim_name_list[i].decode("utf-8")
#         f = open ("../Data/times{}.csv".format(i), 'w')
#         wtr = csv.writer(f, delimiter=',', lineterminator='\n')
#         current_times = [dt for i in range(ntimestep)]
#         wtr.writerow(current_times)
#         f.close()
        
#         # Fix this saving, loading and then saving at some point...
#         np.savetxt("../Data/Stim_raw{}.csv".format(i), 
#                    stim_file[stim][:],
#                    delimiter=",")
#         file = np.genfromtxt("../Data/Stim_raw{}.csv".format(i))
#         writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
#         writer.writerow(file)
        
def prepare_data():
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
        dt = stim_file[stim+'_dt'][:][0]
        f = open ("../Data/times{}.csv".format(i), 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        current_times = [dt for i in range(ntimestep)]
        wtr.writerow(current_times)
        f.close()
        f2 = open("../Data/Stim_raw{}.csv".format(i), 'w')
        wtr2= csv.writer(f2, delimiter=',', lineterminator='\n')
        wtr2.writerow(stim_file[stim][:])
        f2.close()
        
        
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
    volts_fn = vs_fn + str(stim_ind) + '.dat'
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
    os.rename(volts_fn,'../Data/VHotP'+str(real_ind)+'.dat')

    return p_object

def stim_swap(idx, i):
    """
    Stim swap takes 'idx' which is the stim index % 8 and 'i' which is the actual stim idx
    and then deletes the one at 'idx' and replaces it with the stim at i so that 
    neuroGPU reads stims like 13 as stim_raw5 (13 % 8)
    """
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times0.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times0.csv')

def getVolts(idx):
    '''Helper function that gets volts from data and shapes them for a given stim index'''
    fn = vs_fn + str(idx) +  '.h5'    #'.h5' 
    curr_volts =  nrnMreadH5(fn)
    #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    #curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts




def main():
    nstims = 16 # only running first 8 stims
    prepare_data() # setting up directory to have times / stims


    ###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################
    param_values =  np.array(orig_params).reshape(1,-1)
    param_values = np.repeat(param_values, 10, axis=0) # 10 copies of orig params
    print(param_values.shape,  " : param value shape")
       
    ###### CREATE MAPPING ################# 
    #allparams_from_mapping(param_values)
    run_model(0,0)
    data = nrnMread("../Data/VHotP0.dat")
    print(np.max(data))


#     # run neuroGPU -- set up for either ONE gpu or 8
    #if nGpus == 1: 
    for i in range(1,nstims):
        if i != 0:
            # swaps stim0.csv and times0.csv for stimi.csv and timesi.csv
            stim_swap(0,i)
            # run ...  wait is built into run model
            run_model(0,i)

        else: 
            run_model(0,0)

        fig = plt.figure()
        volts = nrnMread("../Data/VHotP{}.dat".format(i))[:10000]
        #target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
        #target_volts_hdf5 = h5py.File(target_volts_path, 'r')
        curr_target_volts =  target_volts_hdf5[i]
        plt.plot(curr_target_volts, color="Red", label="target")
        plt.plot(volts, label="NeuroGPU")
        plt.legend()

        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig('plots/fig{}'.format(i))
        plt.close(fig)







if __name__ == "__main__":
    main()
#     for file in os.listdir('../Data'):
#         if 'h5' in file:
#             print("replacing: ", file)
#             os.remove('../Data/' + file)

#     for file in os.listdir('/tmp/Data'):
#         shutil.move("/tmp/Data/"+ file, "../Data/" + file)