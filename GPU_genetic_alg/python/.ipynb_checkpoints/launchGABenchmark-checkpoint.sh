#!/bin/bash/



source ./benchmark_plan.txt

IFS=',' read -r -a cpuArray <<< "$cpuTrials"
IFS=',' read -r -a offspringArray <<< "$offspring"

if [ $clean = "True" ]; then
    echo cleaning...
    rm runTimeLogs/*.log*
    rm -r profileLogs/*.prof
fi



for offspring_trial in "${offspringArray[@]}" 
    do
    for cpu_trial in "${cpuArray[@]}"
    do
        echo running @ "$cpu_trial" cpus and "$offspring_trial" offspring
        SLURM_JOB_CPUS_PER_NODE=$cpu_trial
        srun python optimize_parameters_genetic_alg.py --offspring_size $offspring_trial  --max_ngen $nGens
        mkdir profileLogs/${cpu_trial}C${N}N_${offspring_trial}O
        mv profileLogs/*.prof profileLogs/${cpu_trial}C${N}N_${offspring_trial}O/
        mv runTimeLogs/runTime.log runTimeLogs/${cpu_trial}C${N}N_${offspring_trial}O.log
    done
done

exit 0