from benchmark_ea.python.optimize_parameters_genetic_alg import main
import multiprocessing as mp


if __name__ == '__main__':
    pool = mp.Pool(10)
    main(pool)