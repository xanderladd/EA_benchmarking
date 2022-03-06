#!/bin/bash

#SBATCH -N 8
#SBATCH -t 00:10:00
#SBATCH -C gpu
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:3
#SBATCH -c 40
#SBATCH -A m3513 


numNodes=8
cd ../
srun -n $numNodes python optimize_parameters_genetic_alg.py --offspring_size 8000  --max_ngen 4 
mv runTimeLogs/runTime.log runTimeLogs/40C${numNodes}N_100O.log

