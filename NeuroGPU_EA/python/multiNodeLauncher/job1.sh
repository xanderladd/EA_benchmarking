#!/bin/bash

#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH -C gpu
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:8
#SBATCH -c 40
#SBATCH -A m3513 


numNodes=2
cd ../
srun -n $numNodes python optimize_parameters_genetic_alg.py --offspring_size 2000  --max_ngen 4 
mv runTimeLogs/runTime.log runTimeLogs/40C${numNodes}N_100O.log

