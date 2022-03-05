#!/bin/bash




source ./strong_benchmark_plan.txt

IFS=',' read -r -a cpuArray <<< "$cpuTrials"
IFS=',' read -r -a offspringArray <<< "$offspring"
IFS=',' read -r -a nodeArray <<< "$N"

echo "Clearing Logs..."
rm logs/*

LSB_JOB_REPORT_MAIL=N

for nnodes in "${nodeArray[@]}" 
    do
    for offspring_trial in "${offspringArray[@]}" 
        do
        for cpu_trial in "${cpuArray[@]}"
        do
            # wait until debug queue is clear
            num_running=`bjobs | wc -l`
            until [ $num_running -lt  1 ]
            do
              echo "sleeping for a min"
              sleep 60
              num_running=`bjobs | wc -l`
            done
            # launch job
            echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes from `pwd`
            if [ -z "$JOBID" ] # no dependency
            then
              name=gen_alg
              logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch  --nodes=${nnodes}  --output ${logpath} batch_run_cl_cori.sh ${offspring_trial} ${n_stim} ${n_sf}) 
              JOBID=$(echo $jid | sed 's/[^0-9]*//g')
            else # dependency
             logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch --dependency=afterany:$JOBID  --nodes=${nnodes} --output ${logpath} batch_run_cl_cori.sh ${offspring_trial}  ${n_stim} ${n_sf} ) # 2 nodes x 8 gpus
              # multiple jobs can depend on a single job
              JOBID=$(echo $jid | sed 's/[^0-9]*//g')
            fi
            
            echo job id is $JOBID 


        done
    done
done

exit 0