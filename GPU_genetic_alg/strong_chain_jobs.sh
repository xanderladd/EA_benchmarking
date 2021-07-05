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
                  logpath=logs/${nnodes}N${cpu_trial}C${offspring_trial}O.log
                  JOBID=$(bsub -nnodes $nnodes -W 30 -P nro106 -alloc_flags "smt4 nvme" -J gen_alg -q debug  -Ep "sh post_exec.sh $cpu_trial ${nnodes} $offspring_trial" -o $logpath "sh batch_run_cl.sh $cpu_trial $offspring_trial" | awk '/is submitted/{print substr($2, 2, length($2)-2);}')
            else # dependency
                 logpath=logs/${nnodes}N${cpu_trial}C${offspring_trial}O.log
                 echo Dependcy: $JOBID
                 JOBID=$(bsub -nnodes $nnodes -W 30 -P nro106 -w  "done($JOBID)" -alloc_flags "smt4 nvme" -J gen_alg -q debug -Ep "sh post_exec.sh $cpus ${nnodes} $offspring_size"  -o $logpath "sh batch_run_cl.sh $cpu_trial $offspring_trial" | awk '/is submitted/{print substr($2, 2, length($2)-2);}')
            fi


        done
    done
done

exit 0