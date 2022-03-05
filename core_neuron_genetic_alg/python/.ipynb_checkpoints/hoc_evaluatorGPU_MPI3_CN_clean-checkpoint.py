import numpy as np
import h5py
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import os
import subprocess
import re
import time
import struct
import bluepyopt.deapext.algorithms as algo
import io
import copy
import csv
import logging
from mpi4py import MPI
import multiprocessing
import csv
import logging
import GPUtil
import sys
from io import StringIO
from config.allen_config import *
from neuron import h
from neuron import coreneuron

from neuron.units import ms, mV

# set up MPI

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()


coreneuron.gpu = True
coreneuron.enable = True
# h.load_file("run_model_cori.hoc")
h.load_file("runModel.hoc")
h.cvode.cache_efficient(1)
h.cvode.use_fast_imem(1)
h.nrnmpi_init()
pc = h.ParallelContext()
# print(f'First: the mpi_id is {pc.id()} ')
# h.load_file("neuron_files/allen_ts/run_model_cori.hoc")
# h.load_file("neuron_files/allen_ts/runModel.hoc")
h.cvode.cache_efficient(True)
h.dt = 0.02
ntimesteps = 10000
tstop = ntimesteps*h.dt
pc.set_maxstep(10 * ms)
param_list = orig_params

logging.info('So should this')
logging.warning('And this, too')
cpu_str = os.environ['SLURM_JOB_CPUS_PER_NODE']
SLURM_CPUS = int(re.search(r'\d+', cpu_str).group() )
nCpus =  SLURM_CPUS#multiprocessing.cpu_count()
logging.info("using nCpus: " + str(nCpus))

class Cell:
    def __init__(self, gid):
#         print(f'BEGIN: the mpi_id is {pc.id()} and gid is {gid}')
        self.hoc_cell = h.cADpyr232_L5_TTPC1_0fb1ca4724()
        self.gid = gid
        pc.set_gid2node(gid, pc.id())
        curr_p = param_list  # TODO ----> #[gid]*gid
        self.update_params(curr_p)
        self.ic = h.IClamp(self.soma[0](.5))
#         self.ic.delay = 100
#         self.ic.dur = 100
        self.ic.amp = 0.5
#         self.ic.delay = 0
        self.ic.dur = 1e9

#         t =  h.Vector(np.cumsum(np.repeat(.02,10000)))
        v = h.Vector().from_python(np.repeat(2,10000))
#         v = h.Vector().from_python(np.genfromtxt('../Data/Stim_raw22.csv', delimiter=','))
        v.play(self.ic, self.ic._ref_amp, True)
#         self.v = h.Vector().record(self.soma[0](.5)._ref_v, sec=self.soma[0])
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp])}
        
        #self.v = v.record(self.soma[0](.5)._ref_v, sec=self.soma[0])
    

#         print(f'INITIALIZE END: the mpi_id is {pc.id()} and gid is {gid}')
    
    def __getattr__(self, name):
        # we don't have it, see if the hoc_cell has it?
        return getattr(self.hoc_cell, name)
    def update_params(self,p):
        
        for curr_sec in self.hoc_cell.all:
            curr_sec.g_pas = p[0]
            curr_sec.e_pas = p[1]
        for curr_sec in self.hoc_cell.axonal:
            curr_sec.gNaTa_tbar_NaTa_t = p[2]
            curr_sec.gK_Tstbar_K_Tst = p[3]
            curr_sec.gNap_Et2bar_Nap_Et2 = p[4]
            curr_sec.gK_Pstbar_K_Pst = p[5]
            curr_sec.gSKv3_1bar_SKv3_1 = p[6]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[7]
            curr_sec.gCa_HVAbar_Ca_HVA = p[8]

        for curr_sec in self.hoc_cell.somatic:
            curr_sec.gSKv3_1bar_SKv3_1 = p[9]
            curr_sec.gCa_HVAbar_Ca_HVA = p[10]
            curr_sec.gNaTs2_tbar_NaTs2_t = p[11]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[12]
            
        for curr_sec in self.hoc_cell.basal:
            curr_sec.gIhbar_Ih = p[13]
        for curr_sec in self.hoc_cell.apical:
            curr_sec.gIhbar_Ih = p[13]
            
            
old_eval = algo._evaluate_invalid_fitness
print("USING nGPUS: ", nGpus)
custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']




            
        
def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

def run_model(): # pass stim as arg and set it with h?
    h.stimFile = '../Data/Stim_raw22.csv'
    h.finitialize(-65)
    h.tstop = tstop
#     pc.nrncore_run("-e %g"%h.tstop, 0)
    pc.psolve(tstop)
    
# def map_run_model(num):
#     run_model()
#     return 1
            

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self, pool, n_stims, n_sfs, sf_module='efel'):
        """Constructor"""
        self.pool = pool
        self.n_stims = n_stims
        self.n_sfs = n_sfs
        self.sf_module = sf_module
#         paramsCSV = '../../../params/params_bbp_full_gpu_tuned_10_based.csv'
#         templateCSV = "../../../params/params_bbp_full_gpu_tuned_10_based.csv"


        data = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        self.opt_ind = np.array(params_opt_ind)
        data = np.array([data[i] for i in self.opt_ind])
        realData = utils.readParamsCSV(templateCSV)
        realOrig = np.array((np.array(realData)[:,1]), dtype=np.float64)
        self.orig_params = orig_params
        self.pmin = np.array((data[:,2]), dtype=np.float64)
        self.pmax = np.array((data[:,3]), dtype=np.float64)
        # make this a function
        self.fixed = {}
        self.params = []
        counter = 0
        for param_idx in range(len(self.orig_params)):
            if param_idx in self.opt_ind:
                idx = np.where(self.opt_ind == param_idx)
                if np.isclose(self.orig_params[param_idx],self.pmin[idx],rtol=.000001) and np.isclose(self.pmin[idx],self.pmax[idx],rtol=.000001):
                    #self.fixed[param_idx] = self.orig_params[param_idx]
                    self.params.append(bpop.parameters.Parameter(self.orig_params[param_idx], bounds=(self.pmin[idx][0]*.999999,self.pmax[idx][0]*1.00001)))
                    counter  +=1 

                else:
                    counter +=1
                    self.params.append(bpop.parameters.Parameter(self.orig_params[param_idx], bounds=(self.pmin[idx][0],self.pmax[idx][0]))) # this indexing is annoying... pmax and pmin weird shape because they are numpy arrays, see idx assignment on line 125... how can this be more clear
            else:
                #self.fixed[param_idx] = self.orig_params[param_idx]
                self.params.append(bpop.parameters.Parameter(66.56, bounds=(40,90)))

        self.weights = opt_weight_list
        self.opt_stim_list = [e for e in opt_stim_name_list if len(e) < 8 ]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
#         if global_rank == 0:
#             #io_start = time.time()
#             #self.target_volts_list = self.make_target_volts(realOrig, self.opt_stim_list)
#             #io_end = time.time()
#             #logging.info("IO:: " + str(io_end - io_start))
#             self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]#np.genfromtxt("targetVolts.csv", delimiter=",")#self.make_target_volts(realOrig, self.opt_stim_list)
#             #realOrig[1] = -72
#             #self.make_target_volts(realOrig, self.opt_stim_list)
#         else:
#             self.target_volts_list = None
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
        #self.target_volts_list = comm.bcast(self.target_volts_list, root=0)
        
        self.dts = []
        self.num_gen = 0
        
    def make_target_volts(self, orig_params, opt_stim_list):
        self.dts = []
        utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts)
        params = orig_params.reshape(-1,1).T
        #params = np.repeat(params, 5 ,axis=0)
        data_volts_list = np.array([])
        allparams = allparams_from_mapping(list(params)) 
        for stimset in range(0,len(opt_stim_list), nGpus):
            p_objects = []
            for gpuId in range(nGpus): 
                if  (gpuId + stimset) >= len(opt_stim_list):
                    break
                if stimset != 0:
                    print("Swapping ", gpuId, gpuId + stimset)
                    if global_rank == 0:
                        stim_swap(gpuId, gpuId + stimset)
                p_objects.append(self.run_model(gpuId, []))
            for gpuId in range(nGpus):
                if  (gpuId + stimset) >= len(opt_stim_list):
                    break 
                p_objects[gpuId].wait()
                if len(data_volts_list) < 1:
                    data_volts_list  = self.getVolts(gpuId)
                else:
                    data_volts_list = np.append(data_volts_list, self.getVolts(gpuId),axis=0)
                print(data_volts_list.shape)
        np.savetxt("targetVoltsGPUNew2.csv", data_volts_list, delimiter=",")
        print(1/0)
        return data_volts_list

        

    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness
        Returns the count of individuals with invalid fitness
        '''
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = [population[0]] + invalid_ind 
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)

    
    def top_SFs(self, run_num=None, max_sfs=0):
        """
        finds scoring functions w/ weight over 50 and pairs them with that stim and sends
        them to mapping function so that we will run so many processes
        Arguments
        --------------------------------------------------------------
        run_num: the number of times neuroGPU has ran for 8 stims,
        keep track of what stims we are picking out score functions for
        """
        all_pairs = []
        if run_num:
            last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
            first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16


            if last_stim > 18:
                last_stim = 18
        else:
            nstims = min(self.n_stims, len(self.opt_stim_list) -  (len(self.opt_stim_list) % nGpus)) # cut off stims
            first_stim = 0
            last_stim = nstims

        print(first_stim,last_stim, "first and last")
        sf_len = len(score_function_ordered_list)
        curr_weights = self.weights[sf_len*first_stim: sf_len*last_stim + sf_len] #get range of sfs for this stim
        stim_correspondance = np.repeat(np.arange(first_stim, last_stim + 1), sf_len) # inclusive
       # TODO make this dynamic to the number of preocessors
        if max_sfs:
            top_inds = curr_weights.argsort()[-(max_sfs):][::-1]
        else:
            top_inds = np.where(curr_weights > 50)[0]
        all_pairs = zip(stim_correspondance[top_inds],top_inds % sf_len) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
        #flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
        return list(all_pairs)
 


    def map_par(self,run_num):
        ''' 
        This function maps out what stim and score function pairs should be mapped to be evaluated in parallel
        first it finds the pairs with the highest weights, the maps them and then adds up the score for each stim
        for every individual.
        
        Parameters
        -------------------- 
        run_num: the amount of times neuroGPU has ran for 8 stims
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,nstims)
        '''
        fxnsNStims = self.top_SFs(run_num=None,max_sfs=self.n_sfs) # 52 stim-sf combinations (stim#,sf#)
        args = []
#         f = h5py.File("../Data/tmp/{}.hdf5".format(global_rank), "w")
        if self.sf_module == 'ipfx':
            stim_idxs = np.unique(np.array([combo[0] for combo in fxnsNStims]))
            for i in stim_idxs:
                argDict = {   
                "i": i,
#                 "j": j,
                "curr_data_volt" : self.data_volts_list[i % nGpus,:,:].astype(np.float32), # can be really big, like 1000,10000
                "curr_target_volt": self.target_volts_list[i].astype(np.float32), # 10k
                "dt": self.dts[i], #TODO: revisit hacking this
                "start": time.time(),
                "curr_stim" : np.genfromtxt(f'../Data/Stim_raw{i}.csv', delimiter=',')
                }
                args.append(argDict)
            scores = sf.callParaIpfx(self.pool,args)
            return scores
            
        
#         print(fxnsNStims, "FXN STIMS)
        for fxnNStim in fxnsNStims:
            i = fxnNStim[0]
            j = fxnNStim[1]
#             if i > 1:
#                 continue
#             f.create_dataset("data_volt{}{}".format(i,j), data=self.data_volts_list[i % nGpus,:,:].astype(np.float32))
#             f.create_dataset("target_volt{}{}".format(i,j), data=self.target_volts_list[i].astype(np.float32))
            
            try:
                
                trasformation_const = h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+score_function_ordered_list[j].decode('ascii')][:]
            except:
                all_scores = os.listdir("/global/cscratch1/sd/zladd/benchmarking/scores")
                choice = np.random.choice(all_scores)
                trasformation_const = h5py.File(scores_path+choice, 'r')['transformation_const_'+score_function_ordered_list[j].decode('ascii')][:]
                

            #TODO: why does this one cause bugs
            if score_function_ordered_list[j].decode('ascii') == 'AP_rise_time':
                continue
            # TODO: refactor traj score so its not sooooo slow
            if "traj" in score_function_ordered_list[j].decode('ascii'):
                continue  
                
            argDict = {   "i": i,
                "j": j,
                "curr_data_volt" : self.data_volts_list[i,:,:].astype(np.float32), # can be really big, like 1000,10000
                "curr_target_volt": self.target_volts_list[i].astype(np.float32), # 10k
                "curr_sf": score_function_ordered_list[j].decode('ascii'),
                "weight": self.weights[len(score_function_ordered_list)*i + j],
                "transformation": trasformation_const,
                "dt": .02, #TODO: revisit hacking this
                "start": time.time(),
            }
            args.append(argDict)
        
        #f.close()
        start = time.time()
        exs = []
        counter = 0
        res = sf.callPara(self.pool,args)
        

        res = np.array(list(res)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
        end = time.time()
        print("all evals took :" , end - start)
        res = res[:,:] 
        prev_sf_idx = 0 
        # look at key of each stim score pair to see how many stims to sum
        #num_selected_stims = len(set([pair[0] for pair in fxnsNStims])) # not always using 8 stims
#         last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
#         first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16
#         if last_stim > 18:
#             last_stim = 18
#         #print(last_stim, first_stim, "last and first")
        first_stim = 0
        last_stim = min(self.n_stims, len(self.opt_stim_list) -  (len(self.opt_stim_list) % nGpus)) # cut off stims that 
        for i in range(first_stim, last_stim):  # iterate stims and sum
            num_sfs = sum([1 for pair in fxnsNStims if pair[0]==i]) #find how many sf indices for this stim
            #print([pair for pair in fxnsNStims if pair[0]==i], "pairs from : ", run_num)
            #print(fxnsNStims[prev_sf_idx:prev_sf_idx+num_sfs], "Currently evaluating")

            if i == 0:
                weighted_sums = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0),(-1,1))
            else:
                #print(prev_sf_idx, "stim start idx", num_sfs, "stim end idx")
                curr_stim_sum = np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0)
                curr_stim_sum = np.reshape(curr_stim_sum, (-1,1))
                weighted_sums = np.append(weighted_sums, curr_stim_sum , axis = 1)
                #print(curr_stim_sum.shape," : cur stim sum SHAPE      ", weighted_sums.shape, ": weighted sums shape")
            prev_sf_idx = prev_sf_idx + num_sfs # update score function tracking index
        return weighted_sums



    
    def getVolts(self,idx):
        '''Helper function that gets volts from data and shapes them for a given stim index'''
        fn = vs_fn + str(idx) + "_" +  str(global_rank) + '.dat'    #'.h5' 
        io_start = time.time()
        curr_volts =  nrnMread(fn)
        io_end = time.time()
        logging.info("IO:: " + str(io_end - io_start))
        #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
        #curr_volts =  nrnMread(fn)
        Nt = int(len(curr_volts)/ntimestep)
        shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
        return shaped_volts
    

    def evaluate_with_lists(self, param_values):
        '''This function overrides the BPOP built in function. It is currently set up to run GPU tasks for each 
        stim in chunks based on number of GPU resources then stacks these results and sends them off to be
        evaluated. It runs concurrently so that while nGpus are busy, results ready for evaluation are evaluated.
        Parameters
        -------------------- 
        param_values: Population sized list of parameter sets to be ran through neruoGPU then scored and evaluated
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,1)
        '''
        global nGpus # can we avoid thiss.....

        self.dts = []
        self.nindv = len(param_values)
        self.data_volts_list = np.array([])

#         if global_rank == 0:
#             ##### TODO: write a function to check for missing data?
#             self.dts = []
#             io_start = time.time()
#             self.dts  = utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts) # reintialize allen stuff for clean run
#             io_end = time.time()
#             logging.info("IO:: " + str(io_end - io_start))
#             # insert negative param value back in to each set
#             #full_params = np.insert(np.array(param_values), 1, orig_params[1], axis = 1)
#             full_params = None
# #             for reinsert_idx in self.fixed.keys():
# #                 self.fixed[reinsert_idx] = -72
# #                 param_values = np.insert(np.array(param_values), reinsert_idx, self.fixed[reinsert_idx], axis = 1)
# #                 full_params = param_values
#             param_values = np.array(param_values)
#             param_values[0] =  np.array(orig_params)
#             param_values[1] =  np.array(orig_params)
#             param_values[:,1] = -param_values[:,1]
            
       
#         else:
#             full_params = None
#             self.dts = None
        ## with MPI we can have different populations so here we sync them up ##
        #full_params = comm.bcast(full_params, root=0)
#         if global_rank != 0:
#             param_values = None 
#         self.dts = comm.bcast(self.dts, root=0)
#         param_values = comm.bcast(param_values, root=0)
#         myChunk = ((len(param_values) // size) * global_rank , (len(param_values) // size) * (global_rank + 1))
#         # this is a bit hacky, just tacks on last ind if we need to because the split isn't great
#         # cleaning
#         if global_rank == size -1:
#             myChunk = (myChunk[0],len(param_values))
#         param_values = param_values[myChunk[0]:myChunk[1],:]
#         # revisit use of this var when you clean
#         my_nindv = len(param_values)

#         self.dts = comm.bcast(self.dts, root=0)
        
#         allparams = allparams_from_mapping(list(param_values))

        param_values = np.array(param_values)

        
        nstims = min(self.n_stims, len(self.opt_stim_list) -  (len(self.opt_stim_list) % nGpus)) # cut off stims that 
        # would make us do inefficient GPU batch
        print("NSTIMS is: ", nstims)
        start_time_sim = time.time()
        p_objects = []
        score = []
        start_times = [] # a bunch of timers
        end_times = []
        eval_times = []
        run_num = 0
        all_volts = []
        all_params = []
#         nGpus = min(6,nstims, len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))

        
        # TODO: can you do this with sys path instead?
        
#         os.chdir("neuron_files/allen_ts/")
        
        ncell = len(param_values)
#         print(np.array(param_values).shape, "PRAM VAL SHAPE")
        gids = range(pc.id(), ncell,pc.nhost()) # round robin
        curr_pvals = np.array(param_values)[gids]
        curr_pvals = np.tile(orig_params, len(param_values)).reshape(len(param_values),-1)
        cells = [Cell(gid) for gid in gids]
        [cell.update_params(p) for p, cell in zip(curr_pvals, cells)]
        print(f'the mpi_id is {pc.id()} and gids are {[gid for gid in gids]}')


        self.data_volts_list = []
        for idx in range(nstims):
           

            start = time.time()
            run_model()
            
            # attempt to mutltithread calls to coreneuron, program pukes
#             self.pool.map(map_run_model,[1,2,3])

            end = time.time()

            print("SIM TOOK: ", end - start)
            start_times.append(start)
            end_times.append(end)
            print(pc.nhost(), "PC N HOST")

            # if pc.id() == 0:

#             print("**************************************************")
            allvs = []
            allvs = [None]*ncell
            allvs = [None]*len(gids)
            for gid in range(len(gids)):
                curr_cell = cells[gid]
#                 print("**************************************************")
    #             curr_cell = pc.gid2cell(gid)
                #curr_vs = curr_cell.v.to_python()
                curr_vs = curr_cell.rd['v'].to_python()

#                 print(f'gid is {gid}')
                #print(curr_vs)
                allvs[gid] = curr_vs

            '''
            next: Fix mpi issue?
            then fix efel assert error
            then iterate for nstims
            '''
#             [print( np.array(allv).shape, "ALL VS") for allv in allvs]
            curr_dvolts = np.array(allvs).reshape(-1,10001)
            curr_dvolts = curr_dvolts[:,:10000]
            print(f'{idx}', "IDX IDX IDX IDX")
            self.data_volts_list.append(curr_dvolts)
#         print( self.data_volts_list.shape)
#         print(self.data_volts_list)
#         if pc.id() == 0:   
#             import matplotlib.pyplot as plt
#             print(self.data_volts_list[0,0,:10])
#             plt.scatter(np.arange(10000),self.data_volts_list[0,0,:])
#             plt.savefig('tst.png')
#             plt.close()
        
        self.data_volts_list = np.array(self.data_volts_list)
        eval_start = time.time()
        score = self.map_par(run_num) # call to parallel eval
        eval_end = time.time()
        eval_times = eval_end - eval_start
        score = np.sum(score, axis=1)
        print(score.shape, 'scores shape!1!!1!1')
        final_score = None
        final_score = np.concatenate(pc.py_allgather(score))
        print(final_score.shape, "FINAL SCORE SHAPE")
        final_score = final_score.reshape(-1,1)
        exit()

        
        # LOGGING
        
        if global_rank  == 0:
            logging.info("neuroGPU: " + str((np.array(end_times) - np.array(start_times))[:nGpus]))
            # ADDED below
            logging.info("neuroGPU ends: " + str((np.array(end_times))))
            logging.info("neuroGPU starts: " + str((np.array(start_times))))

        ####### ONLY USING GPU runtimes that don't intersect with eval
        print("neuroGPU runtimes: " + str(np.array(end_times) - np.array(start_times)))
        print("evaluation took: ", eval_times)
        if global_rank ==0:
            logging.info("evaluation: " + str(eval_times))

        self.num_gen += 1
        eval_end = time.time()
        print("everything took: ", eval_end - start_time_sim)
        if global_rank ==0:
            logging.info("gen size : " + str(self.nindv))
            logging.info("gen" + str(self.num_gen) + " took: " + str(eval_end - start_time_sim))
        print(score.shape, "SCORE SHAPE")
        temp = min(final_score) 
        res = [i for i, j in enumerate(final_score) if j == temp] 
#         print("The Positions of minimum element : " + str(res)) 
#         print("And that is : ", final_score[res], " from ", global_rank)
        print("or is is: ", np.min(final_score))
        print(final_score.shape, "SCORE SHAPE")
        logging.info("score " + str(self.num_gen) +  " : "  + str(final_score[res]))
        print(1/0)

        return final_score
#                 else:
#                     self.targV = self.target_volts_list[i-nGpus+1:i+1] # i = 15, i-nGpus+1 = 8, i+1 = 16 
#                     self.curr_dts = self.dts[i-nGpus+1:i+1] # so therefore we get range(8,16) for dts and targ vs
#                     print("CALLING EVAL")
#                     score = np.append(score,self.map_par(run_num),axis =1) #stacks scores by stim
                    
#                 eval_end = time.time()
#                 eval_times.append(eval_end - eval_start)
#                 if len(all_volts) < 1:
#                     all_volts = self.data_volts_list
#                     all_params = param_values
#                 else:
#                     all_volts = np.append(all_volts,self.data_volts_list,axis=0)
                
#                 self.data_volts_list = np.array([])
#                 run_num += 1
                
#             elif last_batch and last_batch < nGpus:
#                 self.data_volts_list = np.reshape(self.data_volts_list, (nGpus,my_nindv,ntimestep)) # ok
#                 eval_start = time.time()
#                 self.targV = self.target_volts_list[:nstims] # shifting targV and current dts
#                 self.curr_dts = self.dts[:nstims] #  so that parallel evaluator can see just the relevant parts
#                 score = self.map_par(run_num) # call to parallel eval
#                 self.data_volts_list = np.array([])
#                 eval_end = time.time()
#                 eval_times.append(eval_end - eval_start)
        

#         print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
#         if global_rank ==0:
#             logging.info("neuroGPU: " + str((np.array(end_times) - np.array(start_times))[:nGpus]))
#             # ADDED below
#             logging.info("neuroGPU ends: " + str((np.array(end_times))))
#             logging.info("neuroGPU starts: " + str((np.array(start_times))))

#         ####### ONLY USING GPU runtimes that don't intersect with eval
#         print("neuroGPU runtimes: " + str(np.array(end_times) - np.array(start_times)))
#         print("evaluation took: ", eval_times)
#         if global_rank ==0:
#             logging.info("evaluation: " + str(eval_times))

#         self.num_gen += 1
#         print("everything took: ", eval_end - start_time_sim)
#         if global_rank ==0:
#             logging.info("gen size : " + str(self.nindv))
#             logging.info("gen" + str(self.num_gen) + " took: " + str(eval_end - start_time_sim))
#         print(score.shape, "SCORE SHAPE")
#         score = np.reshape(np.sum(score,axis=1), (-1,1))

#         score = comm.gather(score, root=0)
#         score = comm.bcast(score, root=0)
#         final_score = np.concatenate(score)
#         # we have to do this list comprehension because concatenate on gather gives nested list back
#         final_score = np.array([item for sublist in final_score for item in sublist]).reshape(-1,1)
            
        
#         # Minimum element indices in list 
#         # Using list comprehension + min() + enumerate() 
#         temp = min(final_score) 

#         res = [i for i, j in enumerate(final_score) if j == temp] 
#         print("The Positions of minimum element : " + str(res)) 
#         print("And that is : ", final_score[res], " from ", global_rank)
#         print("or is is: ", np.min(final_score))
#         print(final_score.shape, "SCORE SHAPE")
        
#         #if global_rank == 0:
#         logging.info("score " + str(self.num_gen) +  " : "  + str(final_score[res]))

#         return final_score

    

    
    
    
#     def evaluate_with_lists(self, param_values):
#         input_values = np.copy(self.orig_params)
#         for i in range(len(param_values)):
#             curr_opt_ind = self.opt_ind[i]
#             input_values[curr_opt_ind] = param_values[i]
#         lines = []
#         nrn_start = time.time()
#         data_volts_list = run_model(input_values, self.opt_stim_list, self.args.n_stims)
#         nrn_end = time.time()
#         size = int(os.environ['SLURM_NNODES'])
#         global_rank = int(os.environ['SLURM_PROCID'])
#         lines.append(f'neuron took {nrn_end - nrn_start} grank: {global_rank}\n')
            
#         start = time.time()        
#         score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights, self.args.n_stims, self.args.n_sfs)
#         end = time.time()
#         lines.append(f'evaluation took: {end - start} grank: {global_rank} \n')
#         print(lines)
# #         if global_rank < 64:
# #             with open('runTimeLogs/runTime.log', 'a+') as f:
# #                 f.writelines(lines)

# #         ap_tune_score = ap_tune(input_values, self.ap_tune_target, self.ap_tune_stim_name, self.ap_tune_weight)
#         return [score] #+ ap_tune_score] NOT USING AP TUNE

    

    
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness
