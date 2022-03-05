#!/bin/bash

###SBATCH -N ${1}
#SBATCH -t 10:00:00
#SBATCH -C gpu
##SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --gres=gpu:8
#SBATCH -c 10
###SBATCH -o $3
#SBATCH -A m3513 
#SBATCH --exclusive=mcs
#SBATCH --signal=B:USR1@20



numNodes=$SLURM_JOB_NUM_NODES
offspring_size=$2
cores=$1
# divide cores by 8 since we are using 8 nodes
curr_cores=10
n_stim=$3
n_sf=$4

conda deactivate

module load cgpu
module load hpctoolkit
module load openmpi
module unload make
module load cmake/3.20.2
module load hpcsdk
module load openmpi
module load cuda
module load python 

export LD_LIBRARY_PATH=/global/homes/z/zladd/nrn/share/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/install/lib/python:$PYTHONPATH
export PATH=$HOME/install/bin:$PATH
export PATH=/opt/python/3.7.3.2/bin:$PATH


cd python

post_exec_ran=0

post_exec()
{
    nickname=${numNodes}N_${cores}C_${offspring_size}O_${n_stim}S_${n_sf}SF
    count=0
    FILE=runTimeLogs/${nickname}_${count}.log
    nickname_ct=${nickname}_${count}
    while test -f "$FILE"; do
        FILE=runTimeLogs/${nickname}_${count}.log
        nickname_ct=${nickname}_${count}
        ((count=count+1))
    done

    mv runTimeLogs/runTime.log $FILE
    mkdir outputs/$nickname_ct
    mv gpu_utillization.log outputs/${nickname_ct}/gpu_utillization.log
    cp $FILE outputs/${nickname_ct}/
    #cp GPUTilLogs/${cores}C${numNodes}N_80_$count_utilization.log outputs/${cores}C${numNodes}N_80_$count/
    mkdir outputs/${nickname_ct}/profiles
    mv profiles/* outputs/${nickname_ct}/profiles
    mv *.prof outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
    mv ../slurm/${numNodes}N${cores}C${offspring_size}O.out   outputs/${nickname_ct}/

    post_exec_ran=1
    echo "------- DONE ----------"
}

# call your_cleanup_function once we receive USR1 signal
trap 'post_exec' USR1 

numProcs=$(( $numNodes * 8))

real_n_sf=$(( $n_sf * $n_stim))

srun -n $numProcs --export=PATH=/global/project/projectdirs/m2043/$USER/install/bin:$PATH,PYTHONPATH=/global/project/projectdirs/m2043/$USER/install/lib/python:$PYTHONPATH,LD_LIBRARY_PATH=/global/project/projectdirs/m2043/$USER/install/lib:$LD_LIBRARY_PATH,CC=pgcc,CXX=pgc++  x86_64/special -python  -mpi optimize_parameters_genetic_alg.py  --offspring_size ${offspring_size} --max_ngen 50  --n_sfs $real_n_sf --n_stims $n_stim  --n_cpus $curr_cores & wait


#--sf_module ipfx & 

if [ $post_exec_ran -eq 0 ]; then
    post_exec
fi

