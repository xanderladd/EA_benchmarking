#!/bin/bash

#SBATCH -N 3
#SBATCH -t 00:10:00
#SBATCH -C gpu
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:6
#SBATCH -c 40
#SBATCH -A m3513 


numNodes=3
cd ../
srun -n $numNodes python optimize_parameters_genetic_alg.py --offspring_size 3000  --max_ngen 4 
mv runTimeLogs/runTime.log runTimeLogs/40C${numNodes}N_100O.log

