import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import h5py
import pickle as pkl
import bluepyopt as bpop
import nrnUtils
import sys
import matplotlib.backends.backend_pdf




def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

data_dir ='../Data'


def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

with open("/global/cscratch1/sd/zladd/axonstandardized/playground/gen_alg_GPU/python/best_indv_logs/best_indvs_gen_48.pkl", 'rb') \
as f:
    best_ind = pkl.load(f)
best_params = np.array(best_ind[47]) # 48 of these


paramsCSV = '../params/params_bbp_full.csv'
#orig_params = np.array(np.array(nrnUtils.readParamsCSV(paramsCSV))[:,1], dtype=np.float64)
orig_params = h5py.File('../params/params_bbp_full.hdf5', 'r')['orig_full'][0]
print(orig_params.shape)
scores_path = '../scores/'
objectives_file = h5py.File('../objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
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
data = nrnUtils.readParamsCSV(paramsCSV)
#self.opt_ind = params_opt_ind
data = np.array([data[i] for i in params_opt_ind])
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
        pass
#         params.append(bpop.parameters.Parameter(orig_params[param_idx], bounds=(pmin[param_idx],pmax[param_idx])))

for reinsert_idx in fixed.keys():
    best_params = np.insert(np.array(best_params), reinsert_idx, fixed[reinsert_idx], axis = 0)

    
os.chdir("../../genetic_alg2")
sys.path.insert(0,"/global/cscratch1/sd/zladd/axonstandardized/playground/genetic_alg2")
from neuron import h


run_file = './run_model_cori.hoc'
# run_volts_path = '../run_volts_bbp/'
paramsCSV = './params/params_bbp_full.csv'
orig_params = h5py.File('./params/params_bbp_full.hdf5', 'r')['orig_full'][0]
scores_path = './scores/'
objectives_file = h5py.File('./objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = './stims/stims_full.hdf5'
params_opt_ind = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 23]

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

# Number of timesteps for the output volt.
ntimestep = 10000
dt = .02


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


data_volts_list = run_model(list(best_params), opt_stim_list)
target_volts = np.genfromtxt("/global/cscratch1/sd/zladd/axonstandardized/playground/gen_alg_GPU/Data/target_volts_BBP19.csv",delimiter = ',')



# set output pdf name

fig, axs = plt.subplots(nrows=8,ncols=1,figsize=(3,6))
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=.1, right=None, top=1, wspace=None, hspace=None)
for i in range(8):
    axs[i].plot(data_volts_list[i], color="blue", label="NeuroGPU Optimization")
    axs[i].plot(target_volts[i], color="red", label="NEURON target")
    axs[i].legend(loc="upper right", prop={'size': 3})
    axs[i].set_title(opt_stim_list[i], fontsize=8)
    
plt.savefig("/global/cscratch1/sd/zladd/axonstandardized/playground/gen_alg_GPU/GUI/neuroGPU_results0.pdf")


fig, axs = plt.subplots(nrows=10,ncols=1,figsize=(4,8))
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=.1, right=None, top=1, wspace=None, hspace=None)
for i in range(8,18):
    idx = i - 8
    axs[idx].plot(data_volts_list[i], color="blue", label="NeuroGPU Optimization")
    axs[idx].plot(target_volts[i], color="red", label="NEURON target")
    axs[idx].legend(loc="upper right", prop={'size': 3})
    axs[idx].set_title(opt_stim_list[i], fontsize=8)
    
plt.savefig("/global/cscratch1/sd/zladd/axonstandardized/playground/gen_alg_GPU/GUI/neuroGPU_results1.pdf")



