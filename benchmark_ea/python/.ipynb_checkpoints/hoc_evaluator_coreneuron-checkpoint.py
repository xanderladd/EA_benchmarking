import numpy as np
import h5py
import bluepyopt as bpop
import benchmark_ea.python.utils as utils 
import benchmark_ea.python.score_functions as sf
import os
import subprocess
import re
import time
# import struct
import bluepyopt.deapext.algorithms as algo
# import io
import copy
import logging
from mpi4py import MPI
import multiprocessing
from neuron import h
from neuron import coreneuron
from neuron.units import ms, mV
from benchmark_ea.python.core_neuron_L5_TTPC1 import L5_TTPC1_CELL
from benchmark_ea.python.config.coreneuron_config import *
from benchmark_ea.python.config.config import *

 
            
            
        


def run_model(pc): # pass stim as arg and set it with h?
    """
    Function to run h.finitialize and psolve using parallel context from Coreneuron
    """
    h.finitialize(-65)
    h.tstop = tstop
    pc.psolve(tstop)
    


class hoc_evaluator(bpop.evaluators.Evaluator):
    """
    CoreNeuron hoc evaluator class
    Args
        :param pool: (multiprocessing.Pool) pool of cpu processers to use for multiprocessing. This parameter is deprecated on Cori (x86_64) but not on ppc64le.
        :param n_stims: (int) number of stimuli to use in EA
        :param n_sfs: (int) number of score functions to use in EA
        :param sf_module: (optional str) score function module to use in EA.
    """
    def __init__(self, pool, n_stims, n_sfs, sf_module='efel', logger=None):
        
        """Constructor"""
        self.pool = pool
        self.n_stims = n_stims
        self.logger = logger
        self.n_sfs = n_sfs
        self.sf_module = sf_module
        data = utils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        self.opt_ind = np.array(params_opt_ind)
        data = np.array([data[i] for i in self.opt_ind])
        realData = utils.readParamsCSV(paramsCSV)
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
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]        
        self.dts = []
        self.num_gen = 0
        

        
    def my_evaluate_invalid_fitness(toolbox, population):
        '''
        Evaluate the individuals with an invalid fitness
        Returns the count of individuals with invalid fitness
        '''
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = [population[0]] + invalid_ind 
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)
 


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
        # stim-sf combinations (stim#,sf#)
        fxnsNStims = utils.top_SFs(self.n_stims, self.opt_stim_list, self.weights,run_num=None,max_sfs=self.n_sfs) 
        args = []
        
        
        """
        Seperate function
        """
        
        if self.sf_module == 'ipfx':
            stim_idxs = np.unique(np.array([combo[0] for combo in fxnsNStims]))
            for i in stim_idxs:
                argDict = {   
                "i": i,
                "curr_data_volt" : self.data_volts_list[i % nGpus,:,:].astype(np.float32), # can be really big, like 1000,10000
                "curr_target_volt": self.target_volts_list[i].astype(np.float32), # 10k
                "dt": self.dts[i], #TODO: revisit hacking this
                "start": time.time(),
                'logger' : self.logger,
                "curr_stim" : np.genfromtxt(f'../Data/Stim_raw{i}.csv', delimiter=',')
                }
                args.append(argDict)
            scores = sf.callParaIpfx(self.pool,args)
            return scores
            
        
        nstims = min(self.n_stims, len(self.opt_stim_list) -  (len(self.opt_stim_list) % nGpus)) # cut off stims that 

        for fxnNStim in fxnsNStims:
            
            i = fxnNStim[0]
            j = fxnNStim[1]
            if i >= nstims:
                continue
            
   
            #TODO: why does this one cause bugs
            if score_function_ordered_list[j].decode('ascii') == 'AP_rise_time':
                continue
            # TODO: refactor traj score so its not sooooo slow
            if "traj" in score_function_ordered_list[j].decode('ascii'):
                continue  
                
            argDict = {   "i": i,
                "j": j,
                "curr_data_volt" : self.data_volts_list[i,:,:10000].astype(np.float32), # can be really big, like 1000,10000
                "curr_target_volt": self.target_volts_list[i].astype(np.float32), # 10k
                "curr_sf": score_function_ordered_list[j].decode('ascii'),
                "weight": self.weights[len(score_function_ordered_list)*i + j],
                "transformation": None,
                'logger' : self.logger,
                "dt": .02, #TODO: revisit hacking this
                "start": time.time(),
            }
            args.append(argDict)
        
        #f.close()
        start = time.time()
        scores = sf.callPara(self.pool,args, self.logger)
        scores = np.array(list(scores)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
        
        end = time.time()
        print("all evals took :" , end - start)
        # scores = scores[:,:] 
        first_stim = 0
        # cut off stims that we can't or don't want to acomodate
        last_stim = min(self.n_stims, len(self.opt_stim_list)) 
        weighted_sums = utils.weight_scores(first_stim, last_stim, fxnsNStims, scores)
        
        return weighted_sums



   

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

        pc = h.ParallelContext()
        pc.set_maxstep(10 * ms)

        '''
        STEP 1 : if we can -- split up the population across nodes
        
        TODO: move this to utils
        '''
        if size > 1: 
            myChunk = ((len(param_values) // size) * global_rank , (len(param_values) // size) * (global_rank + 1))
            if global_rank == size -1:
                myChunk = (myChunk[0],len(param_values))
            param_values = param_values[myChunk[0]:myChunk[1],:]
            old_pval_len = len(param_values) 
            ncell = len(param_values) * nstims
            # multiply by nstims to stretch out my chunk for all the stims we want to run
            gids = range(myChunk[0]  * nstims , myChunk[1]  * nstims)
            print(f"{gids} are from from rank {global_rank}, with pval len {len(param_values)} and chunk {myChunk}")
        else:
            old_pval_len = len(param_values) 
            ncell = len(param_values) * nstims
            gids = range(pc.id(), ncell,pc.nhost()) # round robin
        
        
        param_values = np.repeat(param_values, nstims)
        curr_pvals = np.array(param_values)#[gids]
        curr_pvals = np.tile(orig_params, ncell).reshape(ncell,-1)
        if self.num_gen == 0:
            '''
            STEP 2 : instantiate L5 TTPC Pyr. Cells
            '''
            self.cells = [L5_TTPC1_CELL(gid, pc, self.num_gen) for gid in gids]
            # refresh start time
            start_time_sim = time.time()
            
        
        '''
        STEP 3 : update cell parameters
        '''
        [cell.update_params(p) for p, cell in zip(curr_pvals, self.cells)]
    
        '''
        STEP 4 : run simulation and time it
        '''

        start = time.time()
        run_model(pc)
        end = time.time()

        print("SIM TOOK: ", end - start)
        start_times.append(start)
        end_times.append(end)

        '''
        STEP 5 : exctract volts
        - note minor bug here for last rank node and 
        '''
        allvs = []
#         allvs = [None]*ncell
        allvs = [None]*len(gids)
        for gid in range(len(gids)):
            if len(self.cells) - 1 < gid: # MONKEY PATCH TO FIX LAST RANK HAVING 52 gids and 40 cells --> why is self.cells too small on last rank sometimes?
                allvs[gid] = allvs[1]
                continue
            curr_cell = self.cells[gid]
#             curr_cell = pc.gid2cell(gid)
            curr_vs = curr_cell.rd['v'].to_python() # grab volatage from cell
    
            allvs[gid] = curr_vs
        '''
        STEP 5 : exctract volts
        '''
        curr_dvolts = np.array(allvs).reshape(nstims, old_pval_len, 10001)
        curr_dvolts = curr_dvolts[:,:10000]
        self.data_volts_list  = curr_dvolts
        
        self.data_volts_list = np.array(self.data_volts_list)
        self.data_volts_list = self.data_volts_list[:,:,:10000]
        eval_start = time.time()
        '''
        STEP 6 : score volts
        '''
        score = self.map_par(run_num) # call to parallel eval
        eval_end = time.time()
        eval_times = eval_end - eval_start
        score = np.sum(score, axis=1)

        '''
        STEP 7 : reduce all volts to rank 1 ... ignore all further steps
        '''
        final_score = None
        if size > 1:
            final_score = np.concatenate(pc.py_allgather(score))
        else:
            final_score = score
        print(final_score.shape, "FINAL SCORE SHAPE")
        final_score = final_score.reshape(-1,1)

        pc.done()
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
        temp = min(final_score) 
        res = [i for i, j in enumerate(final_score) if j == temp] 

        return final_score

# Hack to override EA fitness function to we can use our mapreduce
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness
