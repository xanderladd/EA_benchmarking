#!/bin/bash


while true; 
do jsrun -n 1 -g 6 nvidia-smi --query-gpu=utilization.gpu --format=csv >> gpu_utillization.log; sleep 1; 
done
