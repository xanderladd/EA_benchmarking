#!/bin/bash

# for i in `seq 2`; do
#   srun --exclusive --nodes 1 --ntasks 1 echo "HI I am ${i}" &
# done
# wait

rm -r .ipython_*
rm tst*.txt
srun  -n $SLURM_JOB_NUM_NODES hostname > hosts.txt
launch_nodes="$((${SLURM_NNODES} / 1))" 
for i in `seq $launch_nodes`; do
 sh hack_batch_run_cl_cori.sh $i &
done
wait
