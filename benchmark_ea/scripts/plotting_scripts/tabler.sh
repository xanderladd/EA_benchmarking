#!/bin/bash




cd ../../

python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA


python  -m benchmark_ea.python.plotting.meta_table --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA
 
 
python -m benchmark_ea.python.plotting.meta_table --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python -m benchmark_ea.python.plotting.meta_table --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA