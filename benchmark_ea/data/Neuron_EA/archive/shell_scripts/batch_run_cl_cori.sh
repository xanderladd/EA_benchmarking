#!/bin/bash

###SBATCH -N ${1}
##SBATCH -t 07:00:00
##SBATCH -C haswell
##SBATCH -q regular
## SBATCH --mail-user=zladd@berkeley.edu
###SBATCH -c 42
###SBATCH -o $3
##SBATCH -A m3513 
##SBATCH --exclusive=mcs
##SBATCH --signal=B:USR1@20



numNodes=$5
offspring_size=$2
n_stim=$3
n_sf=$4
log_id=$((1 + $RANDOM % 100))
cores="$((${numNodes} * 64))"



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

    mv runTimeLogs/runTime_${log_id}.log $FILE
    mkdir outputs/$nickname_ct
#     mv gpu_utillization.log outputs/${nickname_ct}/gpu_utillization.log
    cp $FILE outputs/${nickname_ct}/
#     #cp GPUTilLogs/${cores}C${numNodes}N_80_$count_utilization.log outputs/${cores}C${numNodes}N_80_$count/
#     mkdir outputs/${nickname_ct}/profiles
#     mv profiles/* outputs/${nickname_ct}/profiles
    mv *.prof outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
    mv  stdout.log outputs/${nickname_ct}/
    mv slurm/${numNodes}N${cores}C${offspring_size}O.out   outputs/${nickname_ct}/

#     post_exec_ran=1
    echo "------- DONE ----------"
}

# call your_cleanup_function once we receive USR1 signal
trap 'post_exec' USR1 

ulimit -Sn unlimited 


CURRENTDATE=`date +%m_%d_%Y`
startTIME=`date +%T`
custom=''

export IPYTHONDIR=${PWD}/.ipython
export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 200
srun -n ${cores} ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &
sleep 200
CHECKPOINTS_DIR="checkpoints"
mkdir -p ${CHECKPOINTS_DIR}

echo nickname ${nickname}

python  -m cProfile -o ${nickname}.prof  optimize_parameters_genetic_alg.py \
    -vv                                \
    --compile                          \
    --offspring_size=${offspring_size} \
    --max_ngen=50                      \
    --ipyparallel                      \
    --n_sfs $n_sf                     \
    --n_stims $n_stim                \
    --log_id ${log_id} > stdout.log & 
wait

# srun -n $numNodes python -m cProfile -o ${nickname}.prof optimize_parameters_genetic_alg.py --offspring_size ${offspring_size} --max_ngen 1  --n_sfs $n_sf --n_stims $n_stim   & 
# wait

if [ $post_exec_ran -eq 0 ]; then
    post_exec
fi


exit


