import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
import bluepyopt.deapext.utils as utils
import pickle
import time
import numpy as np
from datetime import datetime
import argparse
import sys
import argparse
import time
import textwrap
import os
import cProfile
import shutil
import glob
from mpi4py import MPI
import multiprocessing as mp
import subprocess
import logging
import logging.handlers
import os


try:
    nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))
except:
    nGpus = 1
    
comm = MPI.COMM_WORLD
size = comm.Get_size()
total_rank = comm.Get_rank()
global_rank = comm.Get_rank() // nGpus
local_rank = comm.Get_rank() % nGpus
total_size = comm.Get_size()

gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = utils.update_history_and_hof



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
    parser.add_argument('--continue_cp', type=bool, default=False)
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--simulator', type=str, default='infer',
                        help='what simulator should the optimization use?')
    parser.add_argument('--log_id', type=int, required=False, default=-1,
                        help='id of log')
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
        


def create_cpu_optimizer(args, logger):
    """
    returns configured bluepyopt.optimisations.DEAPOptimisation
    """
    if args.ipyparallel:
        from ipyparallel import Client
        rc = Client(profile=f'benchmark.{args.log_id}')
        #os.getenv('IPYTHON_PROFILE')) 
        logger.debug('Using ipyparallel with %d engines', len(rc))

        lview = rc.load_balanced_view()
        dview = rc.direct_view()
        def mapper(func, it):
            dview.map(os.chdir, [os.getcwd()]*len(rc.ids))
            start_time = datetime.now()
            ret = dview.map_sync(func, it)
            print('Generation took', datetime.now() - start_time)
            if global_rank == 0:
                logging.info(f'Generation took {datetime.now() - start_time}')
                with open(f'tst_{args.log_id}.txt','w') as f:
                    f.writelines('.1')
            return ret

        map_function = mapper
    else:
        map_function = None
        
    import benchmark_ea.python.hoc_evaluator_neuron as hoc_ev
    evaluator = hoc_ev.hoc_evaluator(args, logger)
    seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        map_function=map_function,
        seed=args.seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    print(map_function)

    return opt


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
    args, unk = get_parser().parse_known_args()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats
    if args.log_id > 0:
        filename = filename = f"runTimeLogs/runTime_{args.log_id}.log"
    else:
        filename = filename = f"runTimeLogs/runTime.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    if global_rank == 0 and args.simulator != "NEURON":
        p = subprocess.Popen(["sh","watch_gpu_util.sh"])
    

    
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

    
    
    
    
    
    if args.simulator == "NEURON":
        pool.close()
        opt = create_cpu_optimizer(args, logger)
    else:
        if args.simulator == "NeuroGPU":
            import benchmark_ea.python.hoc_evaluator_neurogpu as hoc_ev
        elif args.simulator == "CoreNeuron":
            import benchmark_ea.python.hoc_evaluator_coreneuron as hoc_ev
        else:
            raise NotImplementedError
        evaluator = hoc_ev.hoc_evaluator(pool, args.n_stims, args.n_sfs, args.sf_module, logger)
        opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        seed=args.seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size,
        continue_cp=args.continue_cp,
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
        
        
def main_with_cprofile(pool):
    datafn = 'main.prof'
    prof = cProfile.Profile()
    retval = prof.runcall(main, pool)
    prof.dump_stats(datafn)
    
    

if __name__ == '__main__':
    pool = mp.Pool(10)
    main(pool)
    if global_rank == 1:
        logger.info("absolute end : " + str(time.time()))
