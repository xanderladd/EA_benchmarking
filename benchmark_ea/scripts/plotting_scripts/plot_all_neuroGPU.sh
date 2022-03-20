#!/bin/bash



benchmark_dir=`pwd`/../slurm_launch/plans
cd ../../data/NeuroGPU_EA/experiments

python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Scales --path regular_outputs 
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Fixed --path regular_outputs 
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Fixed\ and\ Problem\ Scales --path regular_outputs 


python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Scales\ and\ Problem\ Scales --path ipfx_outputs 
# rm ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
# mv ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales\ 8_scale.csv ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Scales\ and\ Problem\ Fixed --path ipfx_outputs 
# rm ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
# mv ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 8_scale.csv ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Fixed\ and\ Problem\ Scales --path ipfx_outputs 
# rm ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
# mv ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 8_scale.csv ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv

python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Scales\ and\ Problem\ Scales --path 8SF_outputs
# rm 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
# mv 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales\ 8_scale.csv 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Scales\ and\ Problem\ Fixed --path 8SF_outputs 
# rm 8SF_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
# mv 8SF_outputs/main_plans_8sf/Compute\ Scales\ and\ Problem\ Fixed\ 8_scale.csv 8SF_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_8sf/Compute\ Fixed\ and\ Problem\ Scales --path 8SF_outputs 
# rm 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
# mv 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 8_scale.csv 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv


python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_6sf/Compute\ Scales\ and\ Problem\ Scales --path 6GPU_outputs 
# rm 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
# mv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales\ 6_scale.csv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_6sf/Compute\ Scales\ and\ Problem\ Fixed  --path 6GPU_outputs 
# rm 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
# mv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 6_scale.csv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/main_plans_6sf/Compute\ Fixed\ and\ Problem\ Scales --path 6GPU_outputs 
# rm 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
# mv 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 6_scale.csv 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv



# not using summit anymore
# python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Scales\ 6 --path summit_outputs 
# rm summit_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
# mv summit_outputs/Compute\ Scales\ and\ Problem\ Scales\ 6_scale.csv summit_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
# python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Scales\ and\ Problem\ Fixed\ 6 --path summit_outputs 
# rm summit_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
# mv summit_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 6_scale.csv summit_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
# python -m benchmark_ea.python.plotting.plot_benchmarking --constraint_file ${benchmark_dir}/Compute\ Fixed\ and\ Problem\ Scales\ 6 --path summit_outputs 
# rm summit_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
# mv summit_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 6_scale.csv summit_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv


# NEURON
# python -m benchmark_ea.python.plotting.plot_benchmarking_basic.py --constraint_file ${bencmark_dir}/Compute\ Scales\ and\ Problem\ Fixed 

# python benchmark_ea.python.plotting.plot_benchmarking_basic.py --constraint_file ${bencmark_dir}/Compute\ Scales\ and\ Problem\ Scales 

# python benchmark_ea.python.plotting.plot_benchmarking_basic.py --constraint_file ${bencmark_dir}/Compute\ Fixed\ and\ Problem\ Scales 
