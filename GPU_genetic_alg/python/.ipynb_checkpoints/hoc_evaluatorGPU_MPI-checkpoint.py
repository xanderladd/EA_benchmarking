import numpy as np
import h5py
import bluepyopt as bpop
import utils
import score_functionsB4CompressArgs as sf
import efel
import pandas as pd
import os
import os.path
import subprocess
import time
import struct
import glob
import ctypes
import matplotlib.pyplot as plt
from extractModel_mappings import   allparams_from_mapping
import bluepyopt.deapext.algorithms as algo
from multiprocessing import Pool
import multiprocessing
import shutil, errno
#from concurrent.futures import ThreadPoolExecutor as Pool
import csv
import ap_tuner as tuner
#os.environ["OMP_NUM_THREADS"] = "80" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MPICH_GNI_FORK_MODE"] = "FULLCOPY" # export MPICH_GNI_FORK_MODE=FULLCOPY
from mpi4py import MPI
from joblib import Parallel, delayed
#from mpi4py.futures import MPIPoolExecutor

####### set up environment variables ##############
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
print([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

# inputFile = open("../../../../../input.txt","r") 
# for line in inputFile.readlines():
#     if "bbp" in line:
from config.allen_config import *
#     elif "allen" in line:
#         from config.allen_config import *

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()
old_eval = algo._evaluate_invalid_fitness

print("USING nGPUS: ", nGpus, " and USING nCPUS: ", nCpus)

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

if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def get_first_zero(stim):
    """Kyung helper function to penalize AP where there should not be one"""
    for i in range(len(stim)-2, -1, -1):
        if stim[i] > 0 and stim[i+1] == 0:
            return i+1
    return None

def check_ap_at_zero(stim_ind, volts):
    """
    Kyung function to check if a volt should be penalized for having an AP before there 
    should be one. Modified to take in "volts" as a list of individuals instead of "volt"
    """
    stim_name = list([e.decode('ascii') for e in opt_stim_name_list])[int(stim_ind)]
    stim = stim_file[stim_name][:]
    first_zero_ind = get_first_zero(stim)
    nindv =volts.shape[0]
    checks = np.zeros(nindv)
    for i in range(nindv):
        volt = volts[i,:]
        if first_zero_ind:
            if np.mean(stim[first_zero_ind:]) == 0:
                first_ind_to_check = first_zero_ind + 1000
                APs = [True if v > 0 else False for v in volt[first_ind_to_check:]]
                if True in APs:
                    #return 400 # threshold parameter that I am still tuning
                    #print("indv:",i, "stim ind: ", stim_ind)
                    checks[i] = 250
    return checks    

def stim_swap(idx, i):
    """
    Stim swap takes 'idx' which is the stim index % 8 and 'i' which is the actual stim idx
    and then deletes the one at 'idx' and replaces it with the stim at i so that 
    neuroGPU reads stims like 13 as stim_raw5 (13 % 8)
    """
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        data = utils.readParamsCSV(paramsCSV)
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
                    print(" opt but fixed idx : ", (self.orig_params[param_idx], self.pmin[idx][0]*.999999,self.pmax[idx][0]*1.00001))
                    counter += 1

                else:
                    print("USING: ", self.opt_ind[idx[0]],self.orig_params[param_idx], (self.pmin[idx],self.pmax[idx]))
                    counter +=1
                    self.params.append(bpop.parameters.Parameter(self.orig_params[param_idx], bounds=(self.pmin[idx][0],self.pmax[idx][0]))) # this indexing is annoying... pmax and pmin weird shape because they are numpy arrays, see idx assignment on line 125... how can this be more clear
            else:
                print("FIXED: ", self.orig_params[param_idx], " idx : ", param_idx)
                self.fixed[param_idx] = self.orig_params[param_idx]

        self.weights = opt_weight_list
        print("NUM FREE PARAMS :", counter)
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        if global_rank == 0:
            self.target_volts_list = np.genfromtxt("targetVolts.csv",delimiter=",")#self.make_target_volts(realOrig, self.opt_stim_list)
        else:
            self.target_volts_list = None
        self.target_volts_list = comm.bcast(self.target_volts_list, root=0)
        self.dts = []
        
    def make_target_volts(self, orig_params, opt_stim_list):
        self.dts = []
        utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts)
        print(orig_params)
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
                    utils.stim_swap(gpuId, gpuId + stimset)
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
        np.savetxt("targetVolts.csv", data_volts_list, delimiter=",")
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
    
    def top_SFs(self):
        """
        finds scoring functions w/ weight over 50 and pairs them with that stim and sends
        them to mapping function so that we will run so many processes
        """
        all_pairs = []
        last_stim = (global_rank+1) * nGpus # ie: 0th rank last_stim = (0+1)*ngpus = ngpus
        first_stim = last_stim - nGpus
        if last_stim > 18:
            last_stim = 18
        print(first_stim,last_stim, "first and last...... rank: ", global_rank)
        for i in range(first_stim, last_stim):#range(first_stim,last_stim):
            sf_len = len(score_function_ordered_list)
            curr_weights = self.weights[sf_len*i: sf_len*i + sf_len] #get range of sfs for this stim
            #top_inds = sorted(range(len(curr_weights)), key=lambda i: curr_weights[i], reverse=True)[:10] #finds top ten biggest weight indices
            top_inds = np.where(curr_weights > 50)[0] # weights bigger than 50 #TODO: maybe this can help glitch
            pairs = list(zip(np.repeat(i,len(top_inds)), [ind for ind in top_inds])) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
            all_pairs.append(pairs)
        flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
        return flat_pairs

    
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
        volts_fn = vs_fn + str(stim_ind +(global_rank*nGpus)) + '.dat'
        if os.path.exists(volts_fn):
            #print("removing ", volts_fn, " from ", global_rank)
            os.remove(volts_fn)
        p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind), str(global_rank)])
        return p_object
    
    # convert the allen data and save as csv
    def convert_allen_data(self):
        """
        Function that sets up our new allen data every run. It reads and writes every stimi
        and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
        """
        for i in range(len(opt_stim_name_list)):
            old_stim = "../Data/Stim_raw{}.csv".format(i)
            old_time = "../Data/times{}.csv".format(i)
            if os.path.exists(old_stim) :
                os.remove(old_stim)
                os.remove(old_time)
        for i in range(len(opt_stim_name_list)):
            stim = opt_stim_name_list[i].decode("utf-8")
            dt = .02 # refactor this later to be read or set to .02 if not configured
            self.dts.append(dt)
            f = open ("../Data/times{}.csv".format(i), 'w')
            wtr = csv.writer(f, delimiter=',', lineterminator='\n')
            current_times = [dt for i in range(ntimestep)]
            wtr.writerow(current_times)
            writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
            writer.writerow(stim_file[stim][:])
    
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
        function that evaluates a stim and score function pair on line 252. Sets i as stim # and sets j as 
        score function #. Evaluates volts for that score function in efel or custom. Normalize scores
         then SENT BACK to MAPPER.
        
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
        j = perm[1]
        counter = 0
        curr_data_volt = self.getVolts(i)#[:,:] 
        curr_target_volt = self.target_volts_list[i]
        curr_sf = score_function_ordered_list[j].decode('ascii')
        curr_weight = self.weights[len(score_function_ordered_list)*i + j]
        transformation = h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
        if curr_weight == 0:
            print("BAD WEIGHTS")
            curr_scores = np.zeros(self.nindv)
        else:
            num_indvs = curr_data_volt.shape[0]
            if curr_sf in custom_score_functions:
                score = [getattr(sf, curr_sf)(curr_target_volt, curr_data_volt[indv,:], self.dts[i]) for indv in range(num_indvs)]
            else:
                score = sf.eval_efel(curr_sf, curr_target_volt, curr_data_volt, self.dts[i])
            curr_scores =  score 
        norm_scores = self.normalize_scores(curr_scores, transformation, i)
        for k in range(len(norm_scores)):
            if np.isnan(norm_scores[k]):
                norm_scores[k] = 1
        return norm_scores * curr_weight
    
    
    def map_par(self):
        ''' 
        This function maps out what stim and score function pairs should be mapped to be evaluated in parallel
        first it finds the pairs with the highest weights, the maps them and then adds up the score for each stim
        for every individual.
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,nstims)
        '''
        comm.Barrier() # so all workers do mapping at the same time
        fxnsNStims = utils.top_SFs(global_rank, score_function_ordered_list, self.weights, nGpus, 0) # 52 stim-sf combinations (stim#,sf#)        

        #multiproc / concurrent future
#         with Pool(nCpus) as p:
#             res = p.map(self.eval_stim_sf_pair, fxnsNStims)
        args = []
        for fxnNStim in fxnsNStims:
            i = fxnNStim[0]
            j = fxnNStim[1]
            argDict = {   "i": i,
                "j": j,
                "data volt" : utils.getVolts(vs_fn,i), # can be really big, like 1000,10000
                "target": self.target_volts_list[i], # 10k
                "curr_sf": score_function_ordered_list[j].decode('ascii'),
                "weight": self.weights[len(score_function_ordered_list)*i + j],
                "transformation": h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+score_function_ordered_list[j].decode('ascii')][:],
                "dt": self.dts[i],
                "start": time.time()
            }
            args.append(argDict)
        exs = []
        counter = 0
        res = sf.callPara(args)
        
        res = np.array(list(res)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
        res = res[:,:] 
        prev_sf_idx = 0 
        # get first and last stim so we cna iterate through them and make their scores from mapping
        last_stim = (global_rank+1) * nGpus # ie: 0th rank last_stim = (0+1)*ngpus = ngpus
        first_stim = last_stim - nGpus
        for i in range(first_stim, last_stim):  # iterate stims and sum
            # ALLEN HACK
            if i > 17:
                continue
            num_curr_sfs = sum([1 for pair in fxnsNStims if pair[0]==i]) #find how many sf indices for this stim
            #print([pair for pair in fxnsNStims if pair[0]==i], "PAIRS for stim :" , i, "from:  ", global_rank)

            AP_penalty = utils.check_ap_at_zero(i, utils.getVolts(vs_fn,i), opt_stim_name_list, stim_file ) # NOT USING
            # sum over score functions for a particular stim
            if i == first_stim:
                weighted_sums = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_curr_sfs, :], axis=0) + AP_penalty ,(-1,1))
            else:
                curr_stim_sum = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_curr_sfs, :], axis=0) + AP_penalty, (-1,1))
                #print(min(curr_stim_sum[:,0]), "MIN SCORE FOR STIM", i)
                weighted_sums = np.append(weighted_sums, curr_stim_sum , axis = 1)
                #print(fxnsNStims[prev_sf_idx:prev_sf_idx+num_curr_sfs], "FXNS BEING SUMMED at ", i)

            prev_sf_idx = prev_sf_idx + num_curr_sfs # update score function tracking index
        return weighted_sums
    
    def getVolts(self,idx):
        '''Helper function that gets volts from data and shapes them for a given stim index'''
        #print( "asking for volts:", idx, " from rank: ", idx)

        fn = vs_fn + str(idx) +  '.dat' #.dat
        curr_volts =  nrnMread(fn)
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
        total_stims = len(opt_stim_name_list)
        stim_range = np.arange(nGpus) + (nGpus * global_rank)
        
        if global_rank == 0:
            ##### TODO: write a function to check for missing data?
            self.dts = []
            full_params = None # spaghetti
            utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts) # reintialize allen stuff for clean run
            # insert negative param value back in to each set
            #full_params = np.insert(np.array(param_values), 1, orig_params[1], axis = 1)
            for reinsert_idx in self.fixed.keys():
                param_values = np.insert(np.array(param_values), reinsert_idx, self.fixed[reinsert_idx], axis = 1)
                full_params = param_values
#             if not full_params:
#                 full_params = param_values
            #full_params[0] =  np.array(orig_params)
            #full_params =  np.tile(np.array(orig_params), len(full_params))
        else:
            full_params = None
            self.dts = None
        ## with MPI we can have different populations so here we sync them up ##
        full_params = comm.bcast(full_params, root=0)

        self.dts = comm.bcast(self.dts, root=0)

        allparams = allparams_from_mapping(list(full_params))
        
    
        self.nindv = len(full_params)
        start_time_sim = time.time()
        p_objects = []
        score = []
        # a bunch of timers
        start_times = []
        end_times = []
        eval_times = []
        
        #start running neuroGPU
        for stim_num in stim_range:
            start_times.append(time.time())
            adjusted_ind = stim_num % nGpus
            p_objects.append(self.run_model(stim_num, []))
            
        # evlauate sets of volts for this rank worker stim range  
        for stim_num in stim_range:
            mod_stim_num = stim_num % (nGpus)
            p_objects[mod_stim_num].wait() #wait to get volts output from previous run then read and stack
            end_times.append(time.time())
            
            if mod_stim_num == nGpus-1:
                eval_start = time.time()
                score = self.map_par() # call to parallel eval
                eval_end = time.time()
                eval_times.append(eval_end - eval_start)
 

        print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
        #print("neuroGPU runtimes: ", np.array(end_times) - np.array(start_times))
        print("evaluation took: ", eval_times)
        print("everything took: ", eval_end - start_time_sim)
        sum_score = np.reshape(np.sum(score,axis=1), (-1,1))

        sendbuf = sum_score
        recvbuf = None
        if global_rank == 0:
            recvbuf = np.empty([size, len(sum_score)], dtype=np.float64)
        comm.Gather(sendbuf, recvbuf, root=0)
        if global_rank == 0:
            #print(np.array(recvbuf).shape, "Rec buff")        
            final_score  = np.sum(recvbuf, axis=0)
        else:
            final_score = None
        final_score = comm.bcast(final_score, root=0)
        #print(np.array(final_score).shape, " : final score shape")
        # Minimum element indices in list 
        # Using list comprehension + min() + enumerate() 
        temp = min(final_score) 
        res = [i for i, j in enumerate(final_score) if j == temp] 
        print("The Positions of minimum element : " + str(res)) 
        #print(final_score, "FINAL SCORE")

        return final_score.reshape(-1,1)

    
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness
