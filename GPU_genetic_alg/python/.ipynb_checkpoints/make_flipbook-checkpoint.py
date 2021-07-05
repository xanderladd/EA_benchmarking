import numpy as np 
import matplotlib.pyplot as plt
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import nrnUtils
import os

objectives_file = h5py.File('./objectives/allen485835016_objectives.hdf5', 'r')
target_volts_path = './target_volts/target_volts_485835016.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

objectives_file = h5py.File('./objectives/allen485835016_objectives_passive.hdf5', 'r')
target_volts_path = './target_volts/target_volts_485835016_passive.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
params_opt_ind = [0,1,4,5, 13,14]

paramsCSV = '../params/params_bbp_full_gpu_tuned_10_based.csv'

param_tbl = np.array(nrnUtils.readParamsCSV(paramsCSV))

orig_params = param_tbl[:,1].astype(np.float64)
labels = param_tbl[:,0].astype(str)

if not os.path.isdir('flipbook/voltage_comparision/'):
    os.makedirs('flipbook/voltage_comparision/')
#objectives_file = h5py.File('./objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
#opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list if len(e.decode('ascii')) < 7]


targetFile = [target_volts_hdf5[s][:] for s in opt_stim_list]

history = h5py.File('genResults.hdf5','r')
volt_keys = np.array([elem for elem in history['best_volts'].keys()]).astype(int)
print(volt_keys)
max_key = np.max(volt_keys)

print('making volt plots ...')
for key in volt_keys:
    if key > 100:
        break
    with PdfPages('flipbook/voltage_comparision/voltage_comparision_{}.pdf'.format(key)) as pdf:
        curr_volts = history['best_volts'][str(key)][:]
        if len(curr_volts) < 10 or curr_volts.shape[0] > len(targetFile):
            continue
        for i in range(curr_volts.shape[0]):
            fig = plt.figure()
            plt.plot(curr_volts[i])
            plt.plot(targetFile[i])
            text = "mean voltage :" + str(np.mean(curr_volts[i])) + 'target_mean_voltage' + str(np.mean(targetFile[i]))
            plt.text(.5, -1.5, text, fontsize=12)
            plt.text(0, 10, str(history['best_params'][str(key)][:]), fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            
        



fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(30,30))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.2)
param_keys = np.array([elem for elem in history['best_params'].keys()]).astype(int)

for key in param_keys:
    rms = []
    curr_params = history['best_params'][str(key)][:]
    if len(curr_params) != len(orig_params):
        tmp = orig_params
        for idx,real_idx in enumerate(params_opt_ind):
            
            tmp[real_idx] = curr_params[idx]
        curr_params = tmp
    
    for param_idx,ax in enumerate(axs.flatten()):
        targ = orig_params[param_idx]
        rms.append((curr_params[param_idx] - targ)**2)
#         if param_idx in params_opt_ind:
#             print(labels[param_idx])
#             print(curr_params[param_idx], "curr v")
#             print(orig_params[param_idx], "orig")
#             print(.1*targ, 10*targ)
#             print(curr_params[param_idx] > .1*targ, curr_params[param_idx] < 10*targ)
        if abs(curr_params[param_idx]) > abs(.1*targ) and abs(curr_params[param_idx]) < abs(10*targ):
            if curr_params[param_idx] > .5 and param_idx < 2:
                print(curr_params, param_idx)
            ax.scatter(curr_params[param_idx], key,zorder=10)
    rms = np.mean(rms)
            
    
for param_idx,ax in enumerate(axs.flatten()):
    ax.scatter(orig_params[param_idx], 0, s=400,color='red',zorder=100)
    ax.set_title(labels[param_idx])
    ax.set_ylabel('generation idx')
    ax.set_xlabel('best param value')
    
with PdfPages('flipbook/param_plot.pdf') as pdf:
    pdf.savefig(fig)
    
    
history.close()