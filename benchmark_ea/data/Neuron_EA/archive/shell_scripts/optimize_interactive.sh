#!/bin/bash



NUM_PROCS=400
counter=5
while :
do
    salloc --nodes 7 --qos interactive --time 04:00:00 --constraint haswell --no-shell
    id=`squeue -u zladd -t R -p interactive | awk -F'[^0-9]+' '{ print $2 }'`
    if [ -z "$id" ]
    then
          echo "\$id is empty!! alloc failed!! exiting!!"
          exit
    fi
    srun -n $NUM_PROCS --jobid $id python -m biophys_optimize.scripts.run_optimize --input_json ./test_input_files/optimizers/test_optimize_input_f13.json > f13_$counter.out
    ((counter=counter+1))
    scancel $id
done