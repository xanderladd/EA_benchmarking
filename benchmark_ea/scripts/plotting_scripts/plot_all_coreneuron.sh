#!/bin/bash


benchmark_dir=`pwd`/../slurm_launch/plans
cd ../../data/CoreNeuron_EA/experiments


# python -m benchmark_ea.python.plotting.plot_benchmarking  --constraint_file ${benchmark_dir}/Compute\ Fixed\ and\ Problem\ Scales

python -m benchmark_ea.python.plotting.plot_benchmarking  --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Scales

# python -m benchmark_ea.python.plotting.plot_benchmarking  --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Fixed