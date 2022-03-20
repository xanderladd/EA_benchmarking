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



# numNodes=$5
# offspring_size=$2
# n_stim=$3
# n_sf=$4
# cores="$((${numNodes} * 64))"


numNodes=1
offspring_size=1
n_stim=3
n_sf=20
cores="$((${numNodes} * 64))"

cd ../../data/Neuron_EA/experiments


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
    cp $FILE outputs/${nickname_ct}/
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

# export IPYTHONDIR=${PWD}/.ipython
# export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

# ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
# sleep 200
# srun -n ${cores} ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &
# sleep 200
# CHECKPOINTS_DIR="checkpoints"
# mkdir -p ${CHECKPOINTS_DIR}

# echo nickname ${nickname}

# profile code
# python  -m cProfile -o ${nickname}.prof  optimize_parameters_genetic_alg.py \
python  -m benchmark_ea.python.optimize_parameters_genetic_alg \
    -vv                                \
    --offspring_size=${offspring_size} \
    --max_ngen=2                      \
    --n_sfs $n_sf                     \
    --n_stims $n_stim                \
    --simulator "NEURON" > stdout.log & 
wait



if [ $post_exec_ran -eq 0 ]; then
    post_exec
fi


exit


