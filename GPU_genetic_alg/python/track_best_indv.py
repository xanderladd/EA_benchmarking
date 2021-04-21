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
import argparse 
#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")
from config.allen_config import *



parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example')

parser.add_argument('--indv_path', type=str, required=False,
                    help='number of individuals in offspring')
args = parser.parse_args()

vs_fn = '/tmp/Data/VHotP'


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
        
        # Fix this saving, loading and then saving at some point...
        np.savetxt("../Data/Stim_raw{}.csv".format(i), 
                   stim_file[stim][:],
                   delimiter=",")
        file = np.genfromtxt("../Data/Stim_raw{}.csv".format(i))
        writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
        writer.writerow(file)
        
        
def run_model(real_ind):
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
    volts_fn = vs_fn + str(real_ind) + '.dat'
    if os.path.exists(volts_fn):
        #print("removing ", volts_fn, " from ", global_rank)
        os.remove(real_ind)
    #!{'../bin/neuroGPU'+str(global_rank),str(stim_ind), str(global_rank)}
    p_object = subprocess.Popen(['../bin/neuroGPU',str(real_ind), str(0)],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,  # <-- redirect stderr to stdout
                    bufsize=1)
    
    with p_object.stdout:
        for line in iter(p_object.stdout.readline, b''):
            print(line),
    p_object.wait()
    print(p_object.stderr)
    print(os.listdir("/tmp/Data/"))

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
    nstims = 18 # only running first 8 stims
    prepare_data() # setting up directory to have times / stims


    if args.indv_path:
        with (open(args.indv_path, "rb")) as openfile:
            all_param_values = pickle.load(openfile)
        param_values = all_param_values[-1]
        param_values = np.insert(np.array(param_values),1, orig_params[1]).reshape(-1,1).T
    else: 
        param_values =  np.array(orig_params).reshape(1,-1)
        param_values = np.repeat(param_values, 10, axis=0) # 10 copies of orig params
        print(param_values.shape,  " : param value shape")
       
    ###### CREATE MAPPING ################# 
    allparams_from_mapping(param_values)
    if os.path.isfile('best_indv_volts.hdf5'):
        os.remove('best_indv_volts.hdf5')
    cache = h5py.File('best_indv_volts.hdf5', 'w')
    run_model(0)
    data = nrnMread("/tmp/Data/VHotP0_0.dat")
    cache.create_dataset('volts/volts0', data=data )
    print(np.max(data))

#     # run neuroGPU -- set up for either ONE gpu or 8
    #if nGpus == 1: 
    for i in range(1,nstims):
        if i != 0:
            # swaps stim0.csv and times0.csv for stimi.csv and timesi.csv
            #stim_swap(0,i)
            # run ...  wait is built into run model
            run_model(i)
            data = nrnMread("/tmp/Data/VHotP{}_0.dat".format(i))
            cache.create_dataset('volts/volts{}'.format(i),data= data )

        else: 
            run_model(0,0)
#     else:
#         for i in range(0,nstims):
#             # run ...  wait is built into run model
#             run_model(i,i)


    cache.close()

if __name__ == "__main__":
    main()
#     for file in os.listdir('../Data'):
#         if 'h5' in file:
#             print("replacing: ", file)
#             os.remove('../Data/' + file)

#     for file in os.listdir('/tmp/Data'):
#         shutil.move("/tmp/Data/"+ file, "../Data/" + file)