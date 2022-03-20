#!/bin/bash


numNodes=8
offspring_size=3000
n_stim=8
n_sf=20
log_id=$1
cores="$((${numNodes} * 64))"

OPENBLAS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

post_exec_ran=0

post_exec()
{
    nickname=${numNodes}N_${cores}C_${offspring_size}O_${n_stim}S_${n_sf}SF
    count="$((${log_id} + 150))"
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
    mv  stdout_$log_id.log outputs/${nickname_ct}/
    mv slurm/${numNodes}N${cores}C${offspring_size}O.out   outputs/${nickname_ct}/

#     post_exec_ran=1
    echo "------- DONE ----------"
}

# call your_cleanup_function once we receive USR1 signal
trap 'post_exec' USR1 

# ulimit -Sn unlimited 


export IPYTHONDIR=${PWD}/.ipython_${log_id}
export IPYTHON_PROFILE=benchmark.${log_id}



extra_sleep="$((${log_id}*10))" # 5-10 extra mins max, 10-extra 20

sleep $extra_sleep
ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 1200
sleep $extra_sleep

# sed -i "${log_id}d" ./hosts.txt
sed -i "1d" ./hosts.txt

# srun -N 1 -n 64 -x ./hosts.txt --exclusive ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &

srun -N $numNodes -n $cores --exclusive ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &
sleep 1200
CHECKPOINTS_DIR="checkpoints"
mkdir -p ${CHECKPOINTS_DIR}

echo nickname ${nickname}



python -m benchmark_ea.python.optimize_parameters_genetic_alg \
    -vv                                \
    --compile                          \
    --offspring_size=${offspring_size} \
    --max_ngen=5                      \
    --ipyparallel                      \
    --n_sfs $n_sf                     \
    --n_stims $n_stim                \
    --simulator "NEURON"               \
    --log_id ${log_id} > stdout_$log_id.log 

post_exec

# if [ $post_exec_ran -eq 0 ]; then
#     post_exec
# fi


exit


