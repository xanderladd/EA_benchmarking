from neuron import h
from neuron import coreneuron
from neuron.units import ms, mV
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

coreneuron.gpu = True
coreneuron.enable = True
h.load_file("runModel.hoc")
h.load_file("template.hoc")
h.cvode.cache_efficient(1)
h.cvode.use_fast_imem(1)
h.nrnmpi_init()
h.cvode.cache_efficient(True)

h.dt = 0.02
ntimestep = 10000
tstop = ntimesteps*h.dt

# print("USING nGPUS: ", nGpus)
custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])

run_volts_path ='../' 
target_volts_path = './target_volts/target_volts_485835016.hdf5'
stims_path = run_volts_path+'/stims/stims_485835016.hdf5'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
objectives_path = './objectives/allen485835016_objectives.hdf5'
allen_param_path = run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5'

target_volts_hdf5 = h5py.File(target_volts_path, 'r')
stim_file = h5py.File(stims_path, 'r')
objectives_file = h5py.File(objectives_path, 'r')
orig_params = h5py.File(allen_param_path, 'r')['orig_full'][0]

scores_path = '../../scores/'
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list']
ap_tune_stim_name = '18'
ap_tune_weight = 0
params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
vs_fn = '../Data/VHotP'

# Number of timesteps for the output volt.
ntimestep = 10000

stim_names = list([e for e in opt_stim_name_list if len(e) < 7])

# orig_params = [0.0000022060, -66.5600000000, 4.0383444556,3.1263250613,0.0047147854,2.8515288117, 5.1781678079, 0.1096452854,0.0005345695, 0.1578655731,0.0016590198, 3.7929209114, 0.0036489355,0.0002609501]



