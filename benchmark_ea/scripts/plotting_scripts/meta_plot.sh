#!/bin/bash

cd ../../

python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA




python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names eFEL IPFX
python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names eFEL IPFX
python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names eFEL IPFX


python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names Cori\ 6\ GPU Summit
python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names Cori\ 6\ GPU Summit
python -m benchmark_ea.python.plotting.meta_plot --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names Cori\ 6\ GPU Summit




# python -m meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA Cori\ CPU

# python -m meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA Cori\ CPU

# python -m meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA Cori\ CPU

