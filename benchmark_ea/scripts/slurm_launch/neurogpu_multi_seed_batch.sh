#!/bin/bash

###SBATCH -N ${1}
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --gres=gpu:8
###SBATCH -c 80
###SBATCH -o $3
#SBATCH -A m3513 
#SBATCH --exclusive=mcs
#SBATCH --signal=B:USR1@20



numNodes=$SLURM_JOB_NUM_NODES
offspring_size=$2
cores=$1
n_stim=$3
n_sf=$4

module purge
module load cgpu
module load gcc cuda cmake/3.21.3 openmpi hpcsdk/20.11 
module load cray-python/3.7.3.2
module load craype


cd python



for seed in {1..10}; do  

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
        touch outputs/${nickname_ct}/seed_${seed}
        #cp GPUTilLogs/${cores}C${numNodes}N_80_$count_utilization.log outputs/${cores}C${numNodes}N_80_$count/
        # mkdir outputs/${nickname_ct}/profiles
        # mv profiles/* outputs/${nickname_ct}/profiles
        # mv *.prof outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
        mv ../slurm/${numNodes}N${cores}C${offspring_size}O.out   outputs/${nickname_ct}/

        post_exec_ran=1
        echo "------- DONE ----------"
    }

    # call your_cleanup_function once we receive USR1 signal
    trap 'post_exec' USR1 

    numProcs=$(( $numNodes * 8))
    real_n_sf=$(( $n_sf * $n_stim))
    curr_cores=10


    srun -n $numProcs -c $curr_cores python -m benchmark_ea.python.optimize_parameters_genetic_alg --offspring_size ${offspring_size} --max_ngen 5  --n_sfs $real_n_sf --n_stims $n_stim  --n_cpus $cores --seed ${seed}  -simulator "NeuroGPU"  > ../slurm/${numNodes}N${cores}C${offspring_size}O.out    & wait

    # --sf_module ipfx 

    if [ $post_exec_ran -eq 0 ]; then
        post_exec
    fi


done