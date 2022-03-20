from neuron import h
from neuron import coreneuron
from neuron.units import ms, mV
import time
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

coreneuron.gpu = True
coreneuron.enable = True
h.load_file("coreneuron_files/runModel.hoc")
# h.load_file("coreneuron_files/template.hoc")
h.cvode.cache_efficient(1)
h.cvode.use_fast_imem(1)
h.nrnmpi_init()
h.cvode.cache_efficient(True)

h.dt = 0.02
ntimestep = 10000
tstop = ntimestep*h.dt

# print("USING nGPUS: ", nGpus)

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])




