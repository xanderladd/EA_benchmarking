#! /bin/bash

module load cgpu

source ./strong_benchmark_plan_pop.txt

IFS=',' read -r -a cpuArray <<< "$cpuTrials"
IFS=',' read -r -a offspringArray <<< "$offspring"
IFS=',' read -r -a nodeArray <<< "$N"
IFS=',' read -r -a n_stims <<< "$n_stims"
IFS=',' read -r -a n_sfs <<< "$n_sfs"

echo "Clearing Slurm..."
rm slurm/*


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
        # launch job
        echo launching @ "$cpu_trial" cpus and "$offspring_trial" offspring and "$nnodes" nodes from `pwd` \
         with $n_stim stims and $n_sf sfs
        if [ -z "$JOBID" ] # no dependency
        then
              name=gen_alg
              logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch  --nodes=${nnodes}  --output ${logpath} job.sh ${offspring_trial} ${n_stim} ${n_sf}) 
              JOBID=$(echo $jid | sed 's/[^0-9]*//g')
        else # dependency
             logpath=slurm/${nnodes}N${cpu_trial}C${offspring_trial}O.out
              jid=$(sbatch --dependency=afterany:$JOBID  --nodes=${nnodes} --output ${logpath} job.sh ${offspring_trial}  ${n_stim} ${n_sf} ) # 2 nodes x 8 gpus
              # multiple jobs can depend on a single job
              #JOBID=$(echo $jid | sed 's/[^0-9]*//g')
        fi
    done
else
    echo weak benchmark plan requires nnodes and offspring size to be same length
fi

#--output ${logpath}