import numpy as np
import h5py
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
import os
import subprocess
import re
import time
import shutil
import struct
import cProfile
import glob
import ctypes
#import matplotlib.pyplot as plt
from extractModel_mappings import   allparams_from_mapping
import bluepyopt.deapext.algorithms as algo
from concurrent.futures import ProcessPoolExecutor as Pool
from concurrent.futures import ProcessPoolExecutor
import sqlite3
import io

#from ray.util.multiprocessing import Pool
import multiprocessing
import csv
import ap_tuner as tuner
import logging
from mpi4py import MPI
import datetime

from config.allen_config import *

#set_start_method("spawn")
#set_start_method("forkserver")
logging.info('Date: ' + str( datetime.datetime.now()))
cpu_str = os.environ['SLURM_JOB_CPUS_PER_NODE']
SLURM_CPUS = int(re.search(r'\d+', cpu_str).group() )
nCpus =  30#SLURM_CPUS#multiprocessing.cpu_count()
logging.info("using nCpus: " + str(nCpus))


#os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
#os.environ["MPICH_GNI_FORK_MODE"] = "FULLCOPY" # export MPICH_GNI_FORK_MODE=FULLCOPY

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])

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
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()


# def adapt_array(arr):
#     """
#     http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
#     """
#     out = io.BytesIO()
#     np.save(out, arr)
#     out.seek(0)
#     return sqlite3.Binary(out.read())

# def convert_array(text):
#     out = io.BytesIO(text)
#     out.seek(0)
#     return np.load(out)


# if os.path.isfile("example{}.db".format(global_rank)):
#     os.remove("example{}.db".format(global_rank))

# # Converts np.array to TEXT when inserting
# sqlite3.register_adapter(np.ndarray, adapt_array)

# # Converts TEXT to np.array when selecting
# sqlite3.register_converter("array", convert_array)

# con = sqlite3.connect("example{}.db".format(global_rank), detect_types=sqlite3.PARSE_DECLTYPES)
# cur = con.cursor()
# cur.execute("create table test (arr array)")
# cur.close()

if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")

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

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        data = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        self.opt_ind = np.array(params_opt_ind)
        data = np.array([data[i] for i in self.opt_ind])
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
                self.fixed[param_idx] = self.orig_params[param_idx]

        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        if global_rank == 0:
            io_start = time.time()
            self.target_volts_list = np.genfromtxt("targetVolts.csv",delimiter=",")
            io_end = time.time()
            logging.info("IO:: " + str(io_end - io_start))
            #self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]#np.genfromtxt("targetVolts.csv", delimiter=",")#self.make_target_volts(realOrig, self.opt_stim_list)
        else:
            self.target_volts_list = None
        self.target_volts_list = comm.bcast(self.target_volts_list, root=0)
        
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
    
    def top_SFs(self, run_num):
        """
        finds scoring functions w/ weight over 50 and pairs them with that stim and sends
        them to mapping function so that we will run so many processes
        Arguments
        --------------------------------------------------------------
        run_num: the number of times neuroGPU has ran for 8 stims,
        keep track of what stims we are picking out score functions for
        """
        all_pairs = []
        last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
        first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16
        if last_stim > 18:
            last_stim = 18
        #print(first_stim,last_stim, "first and last")
        for i in range(first_stim, last_stim):#range(first_stim,last_stim):
            sf_len = len(score_function_ordered_list)
            curr_weights = self.weights[sf_len*i: sf_len*i + sf_len] #get range of sfs for this stim
            #top_inds = sorted(range(len(curr_weights)), key=lambda i: curr_weights[i], reverse=True)[:10] #finds top ten biggest weight indices
            top_inds = np.where(curr_weights > 30)[0] # weights bigger than 50
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
        fxnsNStims = self.top_SFs(run_num) # 52 stim-sf combinations (stim#,sf#)
        args = []
        f = h5py.File("/tmp/{}.hdf5".format(global_rank), "w")

        for fxnNStim in fxnsNStims:
            i = fxnNStim[0]
            j = fxnNStim[1]
            io_start = time.time()
            f.create_dataset("data_volt{}{}".format(i,j), data=self.data_volts_list[i % nGpus,:,:].astype(np.float32))
            f.create_dataset("target_volt{}{}".format(i,j), data=self.target_volts_list[i].astype(np.float32))
            io_end = time.time()
            logging.info("IO:: " + str(io_end - io_start))


            #if global_rank == 0:
#             cur.execute("insert into test (arr) values (?)", (self.data_volts_list[i % nGpus,:,:].astype(np.float32), ))
#             con.commit()

            
            argDict = {   "i": i,
                "j": j,
                #"data volt" : self.data_volts_list[i % nGpus,:,:].astype(np.float32), # can be really big, like 1000,10000
                #"target": self.target_volts_list[i].astype(np.float32), # 10k
                "curr_sf": score_function_ordered_list[j].decode('ascii'),
                "weight": self.weights[len(score_function_ordered_list)*i + j],
                "transformation": h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+score_function_ordered_list[j].decode('ascii')][:],
                "dt": .02,#self.dts[i] #TODO: revisit hacking this
                "start": time.time(),
            }
            args.append(argDict)
        
        f.close()
        print("nCPus", nCpus)
        print("fxn", len(fxnsNStims))
        compression = len(fxnsNStims) // (nCpus - 3)
        print("compression: ", compression)

        compressed_args = [args[i:i+compression] for i in range(0,len(args), compression)]
        print(len(compressed_args))
        print(len(compressed_args[-1]), "len cp args -1")
        print(len(compressed_args[1]), "len cp args 1")
        start = time.time()
        exs = []
        counter = 0
        res = sf.callPara(compressed_args)
        

        res = np.array(list(res)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
        end = time.time()
        print("all evals took :" , end - start)
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



    
    def getVolts(self,idx):
        '''Helper function that gets volts from data and shapes them for a given stim index'''
        fn = vs_fn + str(idx) +  '.dat'    #'.h5' 
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
        self.dts = []
        self.nindv = len(param_values)
        self.data_volts_list = np.array([])

        if global_rank == 0:
            ##### TODO: write a function to check for missing data?
            self.dts = []
            io_start = time.time()
            utils.convert_allen_data(opt_stim_name_list, stim_file, self.dts) # reintialize allen stuff for clean run
            io_end = time.time()
            logging.info("IO:: " + str(io_end - io_start))
            # insert negative param value back in to each set
            #full_params = np.insert(np.array(param_values), 1, orig_params[1], axis = 1)
            full_params = None
            for reinsert_idx in self.fixed.keys():
                param_values = np.insert(np.array(param_values), reinsert_idx, self.fixed[reinsert_idx], axis = 1)
                full_params = param_values
            param_values[0] =  np.array(orig_params)
#             param_values =  np.tile(np.array(orig_params), len(param_values)).reshape( len(param_values), len(orig_params))
        else:
            full_params = None
            self.dts = None
        ## with MPI we can have different populations so here we sync them up ##
        #full_params = comm.bcast(full_params, root=0)
        if global_rank != 0:
            param_values = None 
        param_values = comm.bcast(param_values, root=0)
        param_values = np.array(param_values)
        myChunk = ((len(param_values) // size) * global_rank , (len(param_values) // size) * (global_rank + 1))
        # this is a bit hacky, just tacks on last ind if we need to because the split isn't great
        # cleaning
        if global_rank == size -1:
            myChunk = (myChunk[0],len(param_values))
        param_values = param_values[myChunk[0]:myChunk[1],:]
        # revisit use of this var when you clean
        my_nindv = len(param_values)
        print(myChunk)
        print(np.array(param_values).shape, "PARAM VALUES SHPAE")
        print(my_nindv)

        self.dts = comm.bcast(self.dts, root=0)
        
        allparams = allparams_from_mapping(list(param_values))
        
        nstims = len(self.opt_stim_list) -  (len(self.opt_stim_list) % nGpus) # cut off stims that 
        # would make us do inefficient GPU batch
        print("NSTIMS is: ", nstims)
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
            print("ADDED END TIME for ", i)
            shaped_volts = self.getVolts(idx + nGpus * global_rank)
            if i == 0:
                ap_check_score = utils.check_ap_at_zero(i, shaped_volts, opt_stim_name_list, stim_file).reshape(-1,1)
#                 nospike_score = np.array([utils.noSpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1)
#                 spike_score = np.array([utils.SpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1)
            else:
                ap_check_score = np.append(ap_check_score, utils.check_ap_at_zero(i, shaped_volts, opt_stim_name_list, stim_file).reshape(-1,1),axis=1)
#                 nospike_score = np.append(nospike_score, np.array([utils.noSpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1), axis=1)
#                 spike_score = np.append(nospike_score, np.array([utils.SpikePen(volts, self.target_volts_list[i]) for volts in shaped_volts]).reshape(-1,1), axis=1)

            if idx == 0:
                self.data_volts_list = shaped_volts #start stacking volts
            else:
                self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) 
            #first_batch = i < nGpus # we only evaluate on first batch because we already started neuroGPU
            last_batch  = i == (nstims - 1) # True if we are on last iter
            if not i >= (nstims - nGpus):
                start_times.append(time.time())
                print("ADDED Start TIME for ", i+nGpus, "at ", idx)
                if i != idx:
                    print("replaced dts and stims for ", idx, " with the ones for  ", i)
                    comm.barrier()
                    if global_rank == 0:
                        pass # FIX THIS
                        #stim_swap(idx, i)
                p_objects[idx] = self.run_model(i, []) #ship off job to neuroGPU for next iter
            if idx == nGpus-1:
                self.data_volts_list = np.reshape(self.data_volts_list, (nGpus,my_nindv,ntimestep)) # ok
                eval_start = time.time()
                if i == nGpus-1:
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
#             if last_batch: #end of batch
#                 cutoff = len(self.opt_stim_list) % nGpus
#                 for ii in range(0,cutoff):
#                     p_objects[ii].wait() #wait to get volts output from previous run then read and stack
#                     #end_times.append(time.time())
#                     shaped_volts = utils.getVolts(vs_fn, ii + nGpus * global_rank)
#                     ap_check_score = np.append(ap_check_score, utils.check_ap_at_zero(ii+i, shaped_volts, opt_stim_name_list, stim_file).reshape(-1,1),axis=1)
# #                     nospike_score[:,i+ii] = np.zeros(nospike_score[:,16+ii].shape)#np.array([utils.noSpikePen(volts, self.target_volts_list[16+ii]) for volts in shaped_volts])
# #                     spike_score[:,i+ii] = np.array([utils.SpikePen(volts, self.target_volts_list[16+ii]) for volts in shaped_volts])
#                     if ii == 0:
#                         self.data_volts_list = shaped_volts #start stacking volts
#                     else:
#                         self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) 
                        
#                 self.data_volts_list = np.reshape(self.data_volts_list, (len(self.opt_stim_list) % nGpus,my_nindv,ntimestep))
#                 self.targV = self.target_volts_list[nstims-cutoff:nstims]
#                 self.curr_dts = self.dts[nstims-cutoff:nstims]
#                 score = np.append(score,self.map_par(run_num),axis =1)
                

        print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
        if global_rank ==0:
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
        print("everything took: ", eval_end - start_time_sim)
        if global_rank ==0:
            logging.info("gen size : " + str(self.nindv))
            logging.info("gen" + str(self.num_gen) + " took: " + str(eval_end - start_time_sim))
        print(score.shape, "SCORE SHAPE")
        score = np.reshape(np.sum(score,axis=1), (-1,1))
        ap_check_score = np.sum(ap_check_score, axis=1).reshape(-1,1)
        score += ap_check_score
        
        score = comm.gather(score, root=0)
        score = comm.bcast(score, root=0)
        final_score = np.concatenate(score)
        # we have to do this list comprehension because concatenate on gather gives nested list back
        final_score = np.array([item for sublist in final_score for item in sublist]).reshape(-1,1)
        
        # Minimum element indices in list 
        # Using list comprehension + min() + enumerate() 
        temp = min(final_score) 
        #print(final_score)
        
        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        #con.close()

        res = [i for i, j in enumerate(final_score) if j == temp] 
        print("The Positions of minimum element : " + str(res)) 
        print("And that is : ", final_score[res])
        print(final_score.shape, "SCORE SHAPE")
        return final_score

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness