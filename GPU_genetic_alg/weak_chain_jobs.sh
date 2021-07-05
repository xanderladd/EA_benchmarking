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
        # wait until debug queue is clear
        num_running=`bjobs | wc -l`
        until [ $num_running -lt  1 ]
        do
          echo "sleeping for a min"
          sleep 60
          num_running=`bjobs | wc -l`
        done
        # launch job
        echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes from `pwd` \
         with $n_stim stims and $n_sf sfs
        if [ -z "$JOBID" ] # no dependency
        then
              name=gen_alg
              logpath=logs/${nnodes}N${cpu_trial}C${offspring_trial}O.log
              JOBID=$(bsub -nnodes $nnodes -W 30 -P nro106 -alloc_flags "smt4 nvme" -J gen_alg -q debug  -wa 'USR1' -wt '1'  -o $logpath "sh batch_run_cl.sh $cpu_trial $offspring_trial $n_stim $n_sf" | awk '/is submitted/{print substr($2, 2, length($2)-2);}')
        else # dependency
             logpath=logs/${nnodes}N${cpu_trial}C${offspring_trial}O.log
             echo Dependcy: $JOBID
             JOBID=$(bsub -nnodes $nnodes -W 30 -P nro106 -w  "done($JOBID)" -alloc_flags "smt4 nvme" -J gen_alg -q debug -wa 'USR1' -wt '1'  -o $logpath "sh batch_run_cl.sh $cpu_trial $offspring_trial $n_stim $n_sf" | awk '/is submitted/{print substr($2, 2, length($2)-2);}')
        fi
    done
else
    echo weak benchmark plan requires nnodes and offspring size to be same length
fi
exit 0


# bsub -nnodes 1  -W 5 -P nro106  -q debug  -Ep "sh test_post_exec.sh" "sh sleep.sh"