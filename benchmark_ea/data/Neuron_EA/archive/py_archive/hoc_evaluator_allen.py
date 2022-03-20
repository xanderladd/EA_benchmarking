import numpy as np
import h5py
import os
try:
    os.chdir("neuron_files/allen/")
    from neuron import h
    os.chdir("../../")
except:
    pass
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
import time
import logging
import bluepyopt.deapext.algorithms as algo

from config import *

global gen_count
gen_count = 0
# size = int(os.environ['SLURM_NNODES'])
# global_rank = int(os.environ['SLURM_PROCID'])

def run_model(param_set, stim_name_list, max_stims):
    h.load_file(run_file)
    volts_list = []
    for i,elem in enumerate(stim_name_list):
        if i > max_stims:
            break
        stims_hdf5 = h5py.File(stims_path, 'r')
        curr_stim = stims_hdf5[elem][:]
        total_params_num = len(param_set)
        dt = stims_hdf5[elem+'_dt']
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()        
        volts_list.append(out)
    return np.array(volts_list)

def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list, weights, n_stims, n_sfs):
    def get_fist_zero(stim):
        for i in range(len(stim)-2, -1, -1):
            if stim[i] > 0 and stim[i+1] == 0:
                return i+1
        return None
    def check_ap_at_zero(stim, volt):
        first_zero_ind = get_fist_zero(stim)
        if first_zero_ind:
            if np.mean(stim[first_zero_ind:]) == 0:
                first_ind_to_check = first_zero_ind + 1000
                APs = [True if v > -50 else False for v in volt[first_ind_to_check:]]
                if True in APs:
                    return 400
        return 0
    def eval_function(target, data, function, dt):
        if function in custom_score_functions:
            score = getattr(sf, function)(target, data, dt)
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score
    def normalize_single_score(newValue, transformation):
        # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
        # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
        if newValue > transformation[4]:
            newValue = transformation[4]                                            # Cap newValue to newMax if it is too large
        normalized_single_score = (newValue + transformation[5])/transformation[6]  # Normalize the new score
        if transformation[6] == 0:
            return 1
        return normalized_single_score

    total_score = 0
    top_sf_list = top_SFs(max_sfs=n_sfs, max_stims=n_stims)
    for i,j in top_sf_list:
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt = stims_hdf5[stim_name_list[i]+'_dt'][0]
        curr_stim = stims_hdf5[stim_name_list[i]][:]
        total_score += check_ap_at_zero(curr_stim, curr_data_volt)
        curr_sf = score_function_ordered_list[j].decode('ascii')
        curr_weight = weights[len(score_function_ordered_list)*i + j]
        # not engineered to be a good patch, just engineered to work
            
        if curr_weight == 0:
            curr_score = 0
        else:
            curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
        norm_score = np.array(curr_score)
        if np.isnan(norm_score):
            norm_score = 1
        total_score += norm_score * curr_weight
    return total_score


def top_SFs(max_sfs=0, max_stims=6):
    """
    finds scoring functions w/ weight over 50 and pairs them with that stim and sends
    them to mapping function so that we will run so many processes
    Arguments
    --------------------------------------------------------------
    run_num: the number of times neuroGPU has ran for 8 stims,
    keep track of what stims we are picking out score functions for
    """
    all_pairs = []

    sf_len = len(score_function_ordered_list)

    curr_weights = opt_weight_list[:sf_len*max_stims]
    stim_correspondance = np.repeat(np.arange(max_stims), sf_len) # inclusive
   # TODO make this dynamic to the number of preocessors
    if max_sfs:
        top_inds = curr_weights.argsort()[-(max_sfs):][::-1]
    else:
        top_inds = np.where(curr_weights > 50)[0]
    all_pairs = zip(stim_correspondance[top_inds],top_inds % sf_len) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
    #flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
    return list(all_pairs)


class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self, args, logger=None):
        """Constructor"""
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.args = args
        self.logger = logger
        self.offspring_size = args.offspring_size
        self.log_id = args.log_id
        self.opt_ind = params_opt_ind
        params_ = [params_[i] for i in self.opt_ind]
        self.orig_params = orig_params
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in params_]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        self.weights = opt_weight_list
        self.opt_stim_list = [e for e in opt_stim_name_list if len(e) < 8]

        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
        
    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness
        Returns the count of individuals with invalid fitness
        '''
        global gen_count
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        gen_count += 1
        return len(invalid_ind)


        
    def evaluate_with_lists(self, param_values):
        global gen_count
        # if we are on generation 1 return a random score so we can scratch big gen quickly. 
        if not os.path.isfile(f'tst_{self.log_id}.txt'):
            return [np.random.uniform(0,100)] 
        
        input_values = np.copy(self.orig_params)
        
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
        lines = []
        
        nrn_start = time.time()
        data_volts_list = run_model(input_values, self.opt_stim_list, self.args.n_stims)
        nrn_end = time.time()
        
        start = time.time()        
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights, self.args.n_stims, self.args.n_sfs)
        end = time.time()
        
        #lines.append(f'neuron took {nrn_end - nrn_start} \n') 
        # lines.append(f'evaluation took: {end - start} \n') #grank: {global_rank} \n')


        return [score] 



# import inspect
# import pdb; pdb.set_trace()
# lines = inspect.getsource(algo._evaluate_invalid_fitness)
# print(lines)
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness




