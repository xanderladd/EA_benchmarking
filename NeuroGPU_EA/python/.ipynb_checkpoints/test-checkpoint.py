
def test():
    """
    test
    """
    return 1



def get_parser():
    """
    Get parsed arguemnts from command line. Following arguments are
    :param continue: (bool) should EA continue from checkpoint
    :param checkpoint: (str) path to BluePyOpt formulated checkpoint
    :param offspring_size: (int) number of indivduals to use in EA
    :param max_ngen: (int) number of generations to run to complete EA
    :param n_stims: (int) number of stimuli to use in EA
    :param n_sfs: (int) number of score functions to use in EA
    :param ipyparallel: (bool) use ipyParallel mapping function
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example',
        epilog=textwrap.dedent('''\
The folling environment variables are considered:
    L5PCBENCHMARK_USEIPYP: if set, will use ipyparallel
    IPYTHON_PROFILE: if set, used as the path to the ipython profile
    BLUEPYOPT_SEED: The seed used for initial randomization
        '''))
    parser.add_argument('--continue', type=bool, default=False)
    parser.add_argument('--checkpoint', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--offspring_size', type=int, required=False, default=2,
                        help='number of individuals in offspring')
    parser.add_argument('--max_ngen', type=int, required=False, default=2,
                        help='maximum number of generations')
    parser.add_argument('--n_stims', type=int, required=False, default=1,
                        help='number of stims to optimize over')
    parser.add_argument('--n_sfs', type=int, required=False, default=0,
                        help='number of score functions to use')
    parser.add_argument('--n_cpus', type=int, required=False, default=0,
                        help='number of cpu cores to use')
    parser.add_argument('--sf_module', type=str, required=False, default='efel',
                        help='number of cpu cores to use')
    parser.add_argument('--responses', required=False, default=None,
                        help='Response pickle file to avoid recalculation')
    parser.add_argument('--analyse', action="store_true")
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--hocanalyse', action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--ipyparallel', action="store_true", default=False,
                        help='Use ipyparallel')
    parser.add_argument(
        '--diversity',
        help='plot the diversity of parameters from checkpoint pickle file')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')

    return parser

def my_update(halloffame, history, population):
    """
    Custom update to bluepyopt / DEAP that allows us to save intermediate results
    by calling save_logs
    """
    global gen_counter, cp_freq
    old_update(halloffame, history, population)
    best_indvs.append(halloffame[0])
    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)

    if gen_counter%cp_freq == 0:
        fn = '.pkl'
        save_logs(fn, best_indvs, population)

def save_logs(fn, best_indvs, population):
    """
    Save best individuals as a pickle file in best_indv_logs
    """
    output = open("./best_indv_logs/best_indvs_gen_"+str(gen_counter)+fn, 'wb')
    pickle.dump(best_indvs, output)
    output.close()
    
def my_record_stats(stats, logbook, gen, population, invalid_count):
    """Update the statistics with the new population
    :param logbook: (deap.tools.logbook) DEAP logbook
    """
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)
    logbook = comm.bcast(logbook, root=0)
    if global_rank == 0:
        print('log: ', logbook, '\n')
        output = open("log.pkl", 'wb')
        pickle.dump(logbook, output)
        output.close()

def main(pool):
    """
    Run optimization for NeuroGPU-EA
    :param pool: (multiprocessing.pool) pool of cpu process is created before calling main. 
    This was necessary in ppc64le env. on summit.
    """
    args = get_parser().parse_args()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats
    
    if total_rank == 0:
        # your logging setup
        should_roll_over = os.path.isfile(filename)
        fh = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=15, delay=True)
        if should_roll_over:  # log already exists, roll over!
            print("######### ROLL OVER #######")
            fh.doRollover()

    if total_rank != 0:
        fh = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=15, delay=True)


    comm.barrier()
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("absolute start : " + str(time.time()) + " from rank" + str(global_rank))
    logger.info("nGPUS :" + str(nGpus))



    evaluator = hoc_ev.hoc_evaluator(pool, args.n_stims, args.n_sfs, args.sf_module, logger)
    seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        seed=seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size,
        continue_cp=args.continu,
        cp_filename=args.checkpoint,
        cp_frequency=1)
    if global_rank == 0: # only record root process
        fn = time.strftime("_%d_%b_%Y")
        fn = fn + ".pkl"
        output = open("best_indvs_final"+fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("log"+fn, 'wb')
        pickle.dump(log, output)
        output.close()
        output = open("hst"+fn, 'wb')
        pickle.dump(hst, output)
        output.close()
        output = open("hof"+fn, 'wb')
        pickle.dump(hof, output)
        output.close()
        print ('Hall of fame: ', hof, '\n')
        print ('last log: ', log, '\n')
        print ('History: ', hst, '\n')
        print ('Best individuals: ', best_indvs, '\n')
        
class hoc_evaluator:
    
    def __init__(self, pool, n_stims, n_sfs, sf_module='efel',logger=None):
        """Constructor"""
        return 
        

        

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
    
    def old_top_SFs(self, run_num):
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
           # TODO make this dynamic to the number of preocessors
            top_inds = np.where(curr_weights > 50)[0] # weights bigger than 50
            pairs = list(zip(np.repeat(i,len(top_inds)), [ind for ind in top_inds])) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
            all_pairs.append(pairs)
        flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
        return flat_pairs
    
    def top_SFs(self, run_num, max_sfs=0):
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
        volts_fn = vs_fn + str(stim_ind) + "_" +  str(global_rank) + '.dat'  
        if os.path.exists(volts_fn):
            print("removing ", volts_fn)#, " from ", global_rank)
            # os.remove(volts_fn)
        p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind), str(global_rank)], stdout=subprocess.PIPE)

        time.sleep(.3)
        return p_object
    

            


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
        return 



    
    def getVolts(self,idx):
        '''Helper function that gets volts from data and shapes them for a given stim index'''
        return 

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
        return

    

if __name__ == "__main__":
    test()