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

#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")



run_volts_path = '../'
# paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
# objectives_file = h5py.File('../objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
# orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
# stims_path = run_volts_path+'/stims/allen_data_stims_10000.hdf5'
# target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'

paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based_allzero.csv'
objectives_file = h5py.File('./objectives/allen485835016_objectives_passive.hdf5', 'r')
#orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
stims_path = run_volts_path+'stims/stims_485835016_passive.hdf5'
target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'

paramsCSV = '../params/params_bbp_full_gpu_tuned_10_based.csv'

param_tbl = np.array(nrnUtils.readParamsCSV(paramsCSV))

orig_params = param_tbl[:,1].astype(np.float64)
labels = param_tbl[:,0].astype(str)



scores_path = './scores/allen_scores/'
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
target_volts_hdf5 = [target_volts_hdf5[s][:] for s in opt_stim_name_list]

ap_tune_stim_name = '18'
stim_file = h5py.File(stims_path, 'r')
ntimestep = 10000
vs_fn = '/tmp/Data/VHotP'


nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])

dts = []
utils.convert_allen_data(opt_stim_name_list, stim_file, dts)

if not os.path.isdir('/tmp/Data'):
    os.mkdir('/tmp/Data')

if not os.path.isdir('sweep_plots'):
    os.mkdir('sweep_plots')
    

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


def main():
    nstims = 16 # only running first 8 stims
    transformations = [.1,.5,2,10]

    ###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################
    param_values =  np.array(orig_params).reshape(1,-1)
    param_values = np.repeat(param_values, 3, axis=0)
    param_values[1,14] = 0
    param_values[2,14] = 300
                
    print(param_values.shape,  " : param value shape")
    
    ###### CREATE MAPPING ################# 
    old_params = param_values
    cms = param_values[:,-1:]
    param_values = param_values[:,:-1]
    allparams_from_mapping(param_values,cms=cms)
    


#     # run neuroGPU -- set up for either ONE gpu or 8
    #if nGpus == 1: 
    for i in range(0,nstims):
        run_model(i)

        all_volts = nrnMread("/tmp/Data/VHotP{}_0.dat".format(i))
        plt.plot(all_volts[0:10000], label='original params')
        plt.plot(all_volts[10000:20000], label='cm 0')
        plt.plot(all_volts[20000:30000], label='cm 300')
        plt.legend()
        plt.savefig("cmcheck/cms_{}".format(i))
        plt.close()
        print(1/0)





if __name__ == "__main__":
    main()
#     for file in os.listdir('../Data'):
#         if 'h5' in file:
#             print("replacing: ", file)
#             os.remove('../Data/' + file)

#     for file in os.listdir('/tmp/Data'):
#         shutil.move("/tmp/Data/"+ file, "../Data/" + file)