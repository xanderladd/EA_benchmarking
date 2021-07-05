#!/bin/bash


numNodes=$SLURM_JOB_NUM_NODES
offspring_size=$1
cores=80
n_stim=$2
n_sf=$3



cd ../

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
mv gpu_utillization.log outputs/${nickname_ct}/gpu_utillization.log
cp $FILE outputs/${nickname_ct}/
#cp GPUTilLogs/${cores}C${numNodes}N_80_$count_utilization.log outputs/${cores}C${numNodes}N_80_$count/
mkdir outputs/${nickname_ct}/profiles
mv profiles/* outputs/${nickname_ct}/profiles
mv *.prof outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
echo "------- DONE ----------"