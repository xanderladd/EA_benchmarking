import numpy as np
import h5py
import bluepyopt as bpop
import score_functions as sf
import efel
import pandas as pd
import os
import subprocess
import time
import shutil
import struct
import glob
import ctypes
import matplotlib.pyplot as plt
from extractModel_mappings import   allparams_from_mapping
import bluepyopt.deapext.algorithms as algo
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import csv
import ap_tuner as tuner
import utils
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MPICH_GNI_FORK_MODE"] = "FULLCOPY" # export MPICH_GNI_FORK_MODE=FULLCOPY
from mpi4py import MPI

#inputFile = open("../../../input.txt","r") 
# for line in inputFile.readlines():
#     if "bbp" in line:
#         from config.bbp19_config import *
#     elif "allen" in line:
from config.allen_config import *
if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")


class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        data = utils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        self.opt_ind = params_opt_ind
        data = np.array([data[i] for i in self.opt_ind])
        self.orig_params = orig_params
        self.pmin = np.array((data[:,2]), dtype=np.float64)
        self.pmax = np.array((data[:,3]), dtype=np.float64)
        # make this a function
        self.fixed = {}
        self.params = []
        for param_idx in range(len(orig_params)):
            if np.isclose(self.orig_params[param_idx],self.pmin[param_idx],rtol=.001) and np.isclose(self.pmin[param_idx],self.pmax[param_idx],rtol=.001):
                print(self.orig_params[param_idx], " idx : ", param_idx)
                self.fixed[param_idx] = self.orig_params[param_idx]
            else:
                self.params.append(bpop.parameters.Parameter(self.orig_params[param_idx], bounds=(self.pmin[param_idx],self.pmax[param_idx])))
        
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        self.target_volts_list =  np.array([target_volts_hdf5[s][:] for s in self.opt_stim_list])
        self.dts = []
        

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

    def run_model(self,stim_ind, params):
        """
        Parameters
        -------------------------------------------------------
        stim_ind: index to send as arg to neuroGPU 
        params: DEPRECATED remove
        
        Returns
        ---------------------------------------------------------
        p_object: process object that stops when neuroGPU done
        """
        #volts_fn = vs_fn + str(stim_ind) + '.dat'
        volts_fn = vs_fn + str(stim_ind) + '.h5'
        if os.path.exists(volts_fn):
            os.remove(volts_fn)
        p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
        return p_object
    
    def eval_function(self,target, data, function, dt,i):
        """
        function that sends target and simulated volts to scorefunctions.py so they
        can be evaluated, then adds Kyung AP penalty function at the end
        
        Parameters
        -------------------------------------------------------------
        target: target volt for this stim
        data: set of volts with shape (nindvs, ntimsteps)
        function: string containing score function to use
        dt: dt of the stim as it is a parameter for some sfs
        i: index of the stim
        
        Returns
        ------------------------------------------------------------
        score: scores for each individual in an array corresponding to this score function
        with shape (nindv, 1)
        """
        num_indvs = data.shape[0]
        if function in custom_score_functions:
            score = [getattr(sf, function)(target, data[indv,:], dt) for indv in range(num_indvs)]
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score
    
    def normalize_scores(self,curr_scores, transformation,i):
        '''changed from hoc eval so that it returns normalized score for list of indvs, not just one
        TODO: not sure what transformation[6] does but I changed return statement to fit our 
        dimensions'''
        # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
        # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
        for i in range(len(curr_scores)):
            if curr_scores[i] > transformation[4]:
                curr_scores[i] = transformation[4]        # Cap newValue to newMax if it is too large
        normalized_single_score = (curr_scores + transformation[5])/transformation[6]  # Normalize the new score
        if transformation[6] == 0:
            #return 1
            return np.ones(len(self.nindv)) #verify w/ Kyung
        return normalized_single_score

    def eval_stim_sf_pair(self,perm):
        """ 
        function that evaluates a stim and score function pair on line 252. Sets i as the actual 
        index and and mod_i as it's adjusted index (should get 15th target volt but that will 
        be 7th in the data_volts_list). transform then normalize and then multiply by weight
        and then SENT BACK to MAPPER. uses self. for weights, data_volts, and target volts because
        it is easy information transfer instead of passing arguments into map.
        
        Arguments
        --------------------------------------------------------------------
        perm: pair of ints where first is the stim and second is the score function label index
        to run
        
        Returns
        ---------------------------------------------------------------------
        scores: normalized+weighted scores with the shape (nindv, 1), and sends them back to map
        to be stacked then summed.
        
        """
        i = perm[0]
        mod_i = i % 8
        j = perm[1]
        curr_data_volt = self.data_volts_list[mod_i,:,:]
        curr_target_volt = self.target_volts_list[i]
        curr_sf = score_function_ordered_list[j].decode('ascii')
        curr_weight = self.weights[len(score_function_ordered_list)*i + j]
        transformation = h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
        if curr_weight == 0:
            curr_scores = np.zeros(self.nindv)
        else:
            curr_scores = self.eval_function(curr_target_volt, curr_data_volt, curr_sf, self.dts[i],i)
        norm_scores = self.normalize_scores(curr_scores, transformation,i)
        for k in range(len(norm_scores)):
            if np.isnan(norm_scores[k]):
                norm_scores[k] = 1
        return norm_scores * curr_weight 
    
    
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
        fxnsNStims = utils.top_SFs(run_num, score_function_ordered_list, self.weights, nGpus) # 52 stim-sf combinations (stim#,sf#)
        with Pool(nCpus) as p: # parallel mapping
            res = p.map(self.eval_stim_sf_pair, fxnsNStims)
        res = np.array(list(res)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
        res = res[:,:] 
        prev_sf_idx = 0 
        # look at key of each stim score pair to see how many stims to sum
        #num_selected_stims = len(set([pair[0] for pair in fxnsNStims])) # not always using 8 stims
        last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
        first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16
        if last_stim > 18:
            last_stim = 18
        #print(last_stim, first_stim, "last and first")
        for i in range(first_stim, last_stim):  # iterate stims and sum
            num_sfs = sum([1 for pair in fxnsNStims if pair[0]==i]) #find how many sf indices for this stim
            #print([pair for pair in fxnsNStims if pair[0]==i], "pairs from : ", run_num)
            #print(fxnsNStims[prev_sf_idx:prev_sf_idx+num_sfs], "Currently evaluating")

            if i % nGpus == 0:
                weighted_sums = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0),(-1,1))
            else:
                #print(prev_sf_idx, "stim start idx", num_sfs, "stim end idx")
                curr_stim_sum = np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0)
                curr_stim_sum = np.reshape(curr_stim_sum, (-1,1))
                weighted_sums = np.append(weighted_sums, curr_stim_sum , axis = 1)
                #print(curr_stim_sum.shape," : cur stim sum SHAPE      ", weighted_sums.shape, ": weighted sums shape")
            prev_sf_idx = prev_sf_idx + num_sfs # update score function tracking index
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
        self.dts = []
        utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts) # reintialize allen stuff for clean run
        self.nindv = len(param_values)
        # insert negative param value back in to each set
        for reinsert_idx in self.fixed.keys():
            param_values = np.insert(np.array(param_values), reinsert_idx, self.fixed[reinsert_idx], axis = 1)
            
            
#       ###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################
        #param_values[0] =  np.array(orig_params)#.reshape(1,-1)
#         param_values = np.repeat(param_values, 10, axis=0)
#         print(param_values.shape, "pvals shape!!!!!!!!")
        ###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################

        allparams = allparams_from_mapping(list(param_values)) 
        self.data_volts_list = np.array([])
        nstims = len(self.opt_stim_list)
        nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
        #nstims = 4 # testing
        start_time_sim = time.time()
        p_objects = []
        score = []
        start_times = [] # a bunch of timers
        end_times = []
        eval_times = []
        run_num = 0

        #start running neuroGPU
        for i in range(0, nGpus):
            start_times.append(time.time())
            p_objects.append(self.run_model(i, []))
        # evlauate sets of volts and
        for i in range(0,nstims):
            idx = i % (nGpus)
            p_objects[idx].wait() #wait to get volts output from previous run then read and stack
            end_times.append(time.time())
            shaped_volts = utils.getVolts(vs_fn, idx)
            if i == 0:
                ap_check_score = utils.check_ap_at_zero(i, shaped_volts, opt_stim_name_list, stim_file).reshape(-1,1)
                nospike_score = np.array([utils.noSpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1)
                spike_score = np.array([utils.SpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1)
            else:
                ap_check_score = np.append(ap_check_score, utils.check_ap_at_zero(i, shaped_volts, opt_stim_name_list, stim_file).reshape(-1,1),axis=1)
                nospike_score = np.append(nospike_score, np.array([utils.noSpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1), axis=1)
                spike_score = np.append(nospike_score, np.array([utils.SpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1), axis=1)

            
            if idx == 0:
                self.data_volts_list = shaped_volts #start stacking volts
            else:
                self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) # check this
            first_batch = i < nGpus # we only evaluate on first batch because we already started neuroGPU
            last_batch  = i == (nstims - 1) # True if we are on last iter
            if not first_batch:
                start_times.append(time.time())
                if i != idx:
                    print("replaced dts and stims for ", idx, " with the ones for  ", i)
                    utils.stim_swap(idx, i)
                p_objects[idx] = self.run_model(idx, []) #ship off job to neuroGPU for next iter
            if idx == nGpus-1:
                self.data_volts_list = np.reshape(self.data_volts_list, (nGpus,self.nindv,ntimestep)) # ok
                eval_start = time.time()
                if i == nGpus-1:
                    #print(1/0)
                    self.targV = self.target_volts_list[:nGpus] # shifting targV and current dts
                    self.curr_dts = self.dts[:nGpus] #  so that parallel evaluator can see just the relevant parts
                    score = self.map_par(run_num) # call to parallel eval
                else:
                    self.targV = self.target_volts_list[i-nGpus+1:i+1] # i = 15, i-nGpus+1 = 8, i+1 = 16 
                    self.curr_dts = self.dts[i-nGpus+1:i+1] # so therefore we get range(8,16) for dts and targ vs
                    score = np.append(score,self.map_par(run_num),axis =1) #stacks scores by stim

                    
                eval_end = time.time()
                eval_times.append(eval_end - eval_start)
                self.data_volts_list = np.array([])
                run_num += 1
            #### this part only runs on the last iteration, just to finish off the last two volts and stims
            #### same code as above but only runs three stims instead of 18
            if last_batch: #end of batch
                for ii in range(2):
                    idx = ii % (nGpus)
                    p_objects[idx].wait() #wait to get volts output from previous run then read and stack
                    end_times.append(time.time())
                    shaped_volts = utils.getVolts(vs_fn, idx)
                    ap_check_score[:,16+ii] = utils.check_ap_at_zero(ii+16, shaped_volts, opt_stim_name_list, stim_file)
                    nospike_score[:,16+ii] = np.zeros(nospike_score[:,16+ii].shape)#np.array([utils.noSpikePen(volts, self.target_volts_list[16+ii]) for volts in shaped_volts])
                    spike_score[:,16+ii] = np.array([utils.SpikePen(volts, self.target_volts_list[16+ii]) for volts in shaped_volts])
                    if idx == 0:
                        self.data_volts_list = shaped_volts #start stacking volts
                    else:
                        self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) 
                        
                self.data_volts_list = np.reshape(self.data_volts_list, (2,self.nindv,ntimestep))
                self.targV = self.target_volts_list[16:18]
                self.curr_dts = self.dts[16:18]
                score = np.append(score,self.map_par(run_num),axis =1)

        # TODO: fix timers later
        #print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
        #print("neuroGPU runtimes: ", np.array(end_times) - np.array(start_times))
        #print("evaluation took: ", eval_times)
        print("everything took: ", eval_end - start_time_sim)
        score = np.reshape(np.sum(score,axis=1), (-1,1))
        # here is I am adding penalty
        ap_check_score = np.sum(ap_check_score, axis=1).reshape(-1,1)
        nospike_score = np.sum(nospike_score, axis=1).reshape(-1,1)
        spike_score = np.sum(spike_score, axis=1).reshape(-1,1)

#         print(ap_check_score)
#         print(nospike_score)
#        print(spike_score)

        score = score + ap_check_score + nospike_score #+ spike_score
#        print(score)

        # Minimum element indices in list 
        # Using list comprehension + min() + enumerate() 
        temp = min(score) 
        res = [i for i, j in enumerate(score) if j == temp] 
        print("The Positions of minimum element : " + str(res)) 
        #print(score[0], score[res], " : base score vs best score")
        #testing
#         for i in range(len(score)):
#             print(score[i], ": " + str(i))
            
        print(score.shape, "SCORE SHAPE")
        return score

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness