#!/bin/bash

python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA




python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names eFEL IPFX
python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names eFEL IPFX
python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names eFEL IPFX


python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names Cori\ 6\ GPU Summit
python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names Cori\ 6\ GPU Summit
python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names Cori\ 6\ GPU Summit




# python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA Cori\ CPU

# python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA Cori\ CPU

# python meta_plot.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA Cori\ CPU


