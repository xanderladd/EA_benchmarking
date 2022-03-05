#!/bin/bash

module load cgpu

if [[ $1 == '' ]]; then
    echo no args using strong_benchmark_plan_sf.txt
    source ./strong_benchmark_plan_sf.txt
else
    source ./${1}
fi
IFS=',' read -r -a cpuArray <<< "$cpuTrials"
IFS=',' read -r -a offspringArray <<< "$offspring"
IFS=',' read -r -a nodeArray <<< "$N"
IFS=',' read -r -a n_stims <<< "$n_stims"
IFS=',' read -r -a n_sfs <<< "$n_sfs"

sh waiter.sh


echo "Clearing Logs..."
rm logs/*


LSB_JOB_REPORT_MAIL=N

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
        while [ $num_running -gt 50 ]
        do
            echo "Thresholding number of active jobs to 50"
            sleep 60
            num_running=`squeue -u zladd | wc -l`
        done
        
        FILE=python/outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF*
        if ls $FILE 1> /dev/null 2>&1; then
            echo "python/outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF exists!" 
#             continue
            echo deleting it ... 
            # rm -r python/trash_outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF*
            mv $FILE python/trash_outputs
        else
            echo "novel trial"
        fi
        
        if [ $nnodes -gt 17 ]; then
            echo "${nnodes} is too many nodes, skipping..."
            continue
        fi

        echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes from `pwd` \
         with $n_stim stims and $n_sf sfs
        if [ -z "$JOBID" ] # no dependency
            then
              name=gen_alg
              logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch  --nodes=${nnodes}  --ntasks-per-node 8 -c 10  --output ${logpath} batch_run_cl_cori.sh ${cpu_trial} ${offspring_trial} ${n_stim} ${n_sf}) 
              JOBID=$(echo $jid | sed 's/[^0-9]*//g')
            else # dependency
             logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch --dependency=afterany:$JOBID  --nodes=${nnodes}  --ntasks-per-node 8  -c 10 --output ${logpath} batch_run_cl_cori.sh ${cpu_trial} ${offspring_trial}  ${n_stim} ${n_sf} ) # 2 nodes x 8 gpus
              # multiple jobs can depend on a single job
              JOBID=$(echo $jid | sed 's/[^0-9]*//g')
        fi
        echo JOB ID is : $JOBID
    done
else
    echo weak benchmark plan requires nnodes and offspring size to be same length
fi
exit 0


# bsub -nnodes 1  -W 5 -P nro106  -q debug  -Ep "sh test_post_exec.sh" "sh sleep.sh"