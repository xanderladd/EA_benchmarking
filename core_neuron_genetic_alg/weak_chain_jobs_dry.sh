#!/bin/bash


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

echo "Clearing Logs..."
# rm logs/*

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
            echo "python/outputs/${nnodes}N_${cpu_trial}C_${offspring_trial}O_${n_stim}S_${n_sf}SF exists! exiting" 
            continue
            
        else
            echo "novel trial"
        fi
        
        if [ $nnodes -gt 12 ]; then
            echo "${nnodes} is too many nodes, skipping..."
            continue
        fi
        
        echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes from `pwd` \
         with $n_stim stims and $n_sf sfs
       
    done
else
    echo weak benchmark plan requires nnodes and offspring size to be same length
fi
exit 0


# bsub -nnodes 1  -W 5 -P nro106  -q debug  -Ep "sh test_post_exec.sh" "sh sleep.sh"