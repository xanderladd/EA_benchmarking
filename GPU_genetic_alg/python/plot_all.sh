#!/bin/bash


python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Scales --path regular_outputs 
python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Fixed --path regular_outputs 
python plot_benchmarking.py --constraint_file Compute\ Fixed\ and\ Problem\ Scales --path regular_outputs 


python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Scales\ 8 --path ipfx_outputs 
rm ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
mv ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales\ 8_scale.csv ipfx_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Fixed\ 8 --path ipfx_outputs 
rm ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
mv ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 8_scale.csv ipfx_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Fixed\ and\ Problem\ Scales\ 8 --path ipfx_outputs 
rm ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
mv ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 8_scale.csv ipfx_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv

python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Scales\ 8 --path 8SF_outputs
rm 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
mv 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales\ 8_scale.csv 8SF_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Fixed\ 8 --path 8SF_outputs 
rm 8SF_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
mv 8SF_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 8_scale.csv 8SF_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Fixed\ and\ Problem\ Scales\ 8 --path 8SF_outputs 
rm 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
mv 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 8_scale.csv 8SF_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv


python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Scales\ 6 --path 6GPU_outputs 
rm 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
mv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales\ 6_scale.csv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Fixed\ 6  --path 6GPU_outputs 
rm 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
mv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 6_scale.csv 6GPU_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Fixed\ and\ Problem\ Scales\ 6 --path 6GPU_outputs 
rm 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
mv 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 6_scale.csv 6GPU_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv


python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Scales\ 6 --path summit_outputs 
rm summit_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
mv summit_outputs/Compute\ Scales\ and\ Problem\ Scales\ 6_scale.csv summit_outputs/Compute\ Scales\ and\ Problem\ Scales_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Scales\ and\ Problem\ Fixed\ 6 --path summit_outputs 
rm summit_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
mv summit_outputs/Compute\ Scales\ and\ Problem\ Fixed\ 6_scale.csv summit_outputs/Compute\ Scales\ and\ Problem\ Fixed_scale.csv
python plot_benchmarking.py --constraint_file Compute\ Fixed\ and\ Problem\ Scales\ 6 --path summit_outputs 
rm summit_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv
mv summit_outputs/Compute\ Fixed\ and\ Problem\ Scales\ 6_scale.csv summit_outputs/Compute\ Fixed\ and\ Problem\ Scales_scale.csv