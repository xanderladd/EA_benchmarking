#!/bin/bash


if [[ $1 == '' ]]; then
    echo no benchmarking plan supplied!
    exit 1
else
    source ./${1}
    echo "LAUNCHING ${1}" >> launch_log.log
fi

if [[ $2 != '' ]];  then
    JOBID="$2"
    echo "Launch JOB: $JOBID" >> launch_log.log
else
    echo "WAITING"  >> launch_log.log
    # sh waiter.sh
fi

IFS=',' read -r -a cpuArray <<< "$cpuTrials"
IFS=',' read -r -a offspringArray <<< "$offspring"
IFS=',' read -r -a nodeArray <<< "$N"
IFS=',' read -r -a n_stims <<< "$n_stims"
IFS=',' read -r -a n_sfs <<< "$n_sfs"

echo "Clearing Logs..." >> launch_log.log
rm logs/*




if [ "${#offspringArray[@]}" -eq "${#nodeArray[@]}" ] \
&& [ "${#offspringArray[@]}" -eq "${#n_stims[@]}" ] \
&& [ "${#offspringArray[@]}" -eq "${#n_sfs[@]}" ]; then
    cpu_trial=42 # set cpus to 42 each time
    for i in "${!nodeArray[@]}"; do
        n_sf="${n_sfs[i]}"
        n_stim="${n_stims[i]}"
        nnodes="${nodeArray[i]}"
        offspring_trial="${offspringArray[i]}"       
        # launch job
        num_running=`squeue -u zladd | wc -l`
       
        FILE=python/outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF*

        if ls $FILE 1> /dev/null 2>&1 && [ "$novel" == 'True' ]; then
            echo "python/outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF exists! continuing"  >> launch_log.log
            # deletes any trials with that name
            if [ "$delete" == 'True' ]; then
                echo deleting it ...  >> launch_log.log
                rm -r python/trash_outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF*
                mv $FILE python/trash_outputs
            else 
                continue # you are in novel mode, but you don't want to delete
            fi
        else
            echo "novel trial" >> launch_log.log
        fi
        
        if [ $nnodes -gt 17 ]; then
            echo "${nnodes} is too many nodes, skipping..."
            continue
        fi
        
        # interactive mode
        launchNodes="$((${nnodes} + 1))"
        salloc --nodes $launchNodes --qos interactive --time 04:00:00 --constraint haswell --exclusive=mcs  --signal=USR1@20  --no-shell
        id=`squeue -u zladd -t R -p interactive | awk -F'[^0-9]+' '{ print $2 }' | xargs`
        
        
        if [ -z "$id" ]
        then
              echo "\$id is empty!! alloc failed!! exiting!!"
              exit
        fi
        export SLURM_JOB_NUM_NODES=$nnodes
        
        echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes.. depends on $JOBID from `pwd` \
         with $n_stim stims and $n_sf sfs >> launch_log.log
        

        numNodes=$nnodes
        offspring_size=$offspring_trial
        # cores=$1
        cores="$((${numNodes} * 64))"
        post_exec_ran=0
        nickname=${numNodes}N_${cores}C_${offspring_size}O_${n_stim}S_${n_sf}SF

        post_exec() {
            nickname=${numNodes}N_${cores}C_${offspring_size}O_${n_stim}S_${n_sf}SF
            count=0
            FILE=runTimeLogs/${nickname}_${count}.log
            nickname_ct=${nickname}_${count}
            while test -f "$FILE";  do
                FILE=runTimeLogs/${nickname}_${count}.log
                nickname_ct=${nickname}_${count}
                count="$((${count} + 1))"
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

        export IPYTHONDIR=${PWD}/.ipython
        export IPYTHON_PROFILE=benchmark.${id}
        echo "${IPYTHON_PROFILE} ipython prof"

        ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
        sleep 200
        
        srun -n  ${cores} --jobid $id --exclusive ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &
        sleep 200
        CHECKPOINTS_DIR="checkpoints"
        mkdir -p ${CHECKPOINTS_DIR}

        echo nickname ${nickname}
        # srun -n 1 -c 1  --jobid $id --export=IPYTHON_PROFILE=$IPYTHON_PROFILE, IPYTHONDIR=$IPYTHONDIR 
        python  -m cProfile -o ${nickname}.prof  optimize_parameters_genetic_alg.py \
            -vv                                \
            --compile                          \
            --offspring_size=${offspring_size} \
            --max_ngen=3                      \
            --ipyparallel                      \
            --n_sfs $n_sf                     \
            --n_stims $n_stim > stdout.log 

        # srun -n $numNodes python -m cProfile -o ${nickname}.prof optimize_parameters_genetic_alg.py --offspring_size ${offspring_size} --max_ngen 1  --n_sfs $n_sf --n_stims $n_stim   & 
        # wait

        if [ $post_exec_ran -eq 0 ]; then
            post_exec
        fi

        echo JOB ID is : $JOBID
        scancel $id
        exit
    done
else
    echo weak benchmark plan requires nnodes and offspring size to be same length
    exit 1
fi

echo "FINAL JOB ID is : $JOBID"
exit 0
