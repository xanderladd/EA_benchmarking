#!/bin/bash

# regular
bencmark_dir=`pwd`/../slurm_launch/plans
cd ../../data/Neuron_EA/experiments

python -m benchmark_ea.python.plotting.plot_benchmarking_basic --constraint_file ${bencmark_dir}/Compute\ Scales\ and\ Problem\ Fixed 

python -m benchmark_ea.python.plotting.plot_benchmarking_basic --constraint_file ${bencmark_dir}/Compute\ Scales\ and\ Problem\ Scales 

python -m benchmark_ea.python.plotting.plot_benchmarking_basic --constraint_file ${bencmark_dir}/Compute\ Fixed\ and\ Problem\ Scales 