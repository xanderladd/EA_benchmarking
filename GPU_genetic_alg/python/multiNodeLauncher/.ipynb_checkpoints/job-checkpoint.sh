#!/bin/bash

###SBATCH -N ${1}
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --gres=gpu:6
#SBATCH -c 80
###SBATCH -o $3
#SBATCH -A m3513 
#SBATCH --signal=B:USR1@20



numNodes=$SLURM_JOB_NUM_NODES
offspring_size=$1
cores=80
n_stim=$2
n_sf=$3


cd ../

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
    post_exec_ran=1
    echo "------- DONE ----------"
}

# call your_cleanup_function once we receive USR1 signal
trap 'post_exec' USR1 

srun -n $numNodes python -m cProfile -o ${nickname}.prof optimize_parameters_genetic_alg.py --offspring_size ${offspring_size} --max_ngen 1000  --n_sfs $n_sf --n_stims $n_stim & 
wait

if [ $post_exec_ran -eq 0 ]; then
    post_exec
fi

