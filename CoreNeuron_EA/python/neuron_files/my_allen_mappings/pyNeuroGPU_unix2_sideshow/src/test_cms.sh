#!/bin/bash

make
status=1
cd ../bin/
if [[ "$status" == 1 ]]; then
srun -n 1 ./neuroGPU "0" "0"
fi
