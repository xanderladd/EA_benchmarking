#!/bin/bash

echo postexec dir `pwd`
cd  /gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/


numNodes=$2
offspring_size=$3
cores=$1
n_stim=$4
n_sf=$5

echo $1 $2 $3 $4 $5 are the args

nickname=${numNodes}N_${cores}C_${offspring_size}O_${n_stim}S_${n_sf}SF



FILE=python/runTimeLogs/${nickname}.log
count=0
nickname_ct=${nickname}_${count}
echo $nickname_ct is nickname

while test -f "$FILE"; do
    FILE=python/runTimeLogs/${nickname}_${count}.log
    nickname_ct=${nickname}_${count}
    ((count=count+1))
done

mv python/runTimeLogs/runTime.log $FILE
mkdir python/outputs/$nickname_ct
mv python/gpu_utillization.log outputs/${nickname_ct}/gpu_utillization.log
cp $FILE python/outputs/${nickname_ct}/
#cp GPUTilLogs/${cores}C${numNodes}N_80_$count_utilization.log outputs/${cores}C${numNodes}N_80_$count/
mkdir python/outputs/${nickname_ct}/profiles
mv python/profiles/* python/outputs/${nickname_ct}/profiles
mv python/*.prof python/outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
mv logs/* python/outputs/${nickname_ct}/ 
echo "------- DONE ----------"