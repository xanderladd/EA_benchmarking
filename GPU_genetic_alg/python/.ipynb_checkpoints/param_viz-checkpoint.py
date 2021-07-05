import numpy as np 
import matplotlib.pyplot as plt
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import nrnUtils
import os
import re
import pickle as pkl

#objectives_file = h5py.File('./objectives/allen485835016_objectives_passive.hdf5', 'r')
#target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'
#target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
params_opt_ind = [0,1,4,5, 13,14]
paramsCSV = '../params/params_bbp_full_gpu_tuned_10_based.csv'
param_tbl = np.array(nrnUtils.readParamsCSV(paramsCSV))
orig_params = param_tbl[:,1].astype(np.float64)
labels = param_tbl[:,0].astype(str)
#opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
#opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list if len(e.decode('ascii')) < 7]


if __name__ == "__main__":

    fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(30,30))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.2)
    param_files = os.listdir("best_indv_logs")
    param_files = [file for file in param_files if '.pkl' in file]


    for file in param_files:
        pkl_num = int(re.findall(r'\d+',file)[0])
        rms = []
        fn = os.path.join("best_indv_logs",file)

        with open(fn,'rb') as f:
            curr_params = pkl.load(f)[-1]
        
        # fill in params if we are not optimizing full params
        if len(curr_params) != len(orig_params):
            tmp = orig_params
            for idx,real_idx in enumerate(params_opt_ind):
                tmp[real_idx] = curr_params[idx]
            curr_params = tmp

        for param_idx,ax in enumerate(axs.flatten()):
            targ = orig_params[param_idx]
            rms.append((curr_params[param_idx] - targ)**2)
            # remove param values that are outside base 10 since that make plot hard to read
            if abs(curr_params[param_idx]) > abs(.1*targ) and abs(curr_params[param_idx]) < abs(10*targ):
    #             if curr_params[param_idx] > .5 and param_idx < 2:
    #                 print(curr_params, param_idx)
                print(file)
                ax.scatter(curr_params[param_idx],pkl_num,zorder=10)
        rms = np.mean(rms)


    for param_idx,ax in enumerate(axs.flatten()):
        ax.scatter(orig_params[param_idx], 0, s=400,color='red',zorder=100)
        ax.set_title(labels[param_idx])
        ax.set_ylabel('generation idx')
        ax.set_xlabel('best param value')

    with PdfPages('param_plot.pdf') as pdf:
        pdf.savefig(fig)

    