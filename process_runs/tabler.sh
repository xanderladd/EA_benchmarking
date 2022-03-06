#!/bin/bash


python meta_table.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Fixed --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA
 
 
python meta_table.py --constraint_file neuron_genetic_alg/plans/Compute\ Fixed\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA

python meta_table.py --constraint_file neuron_genetic_alg/plans/Compute\ Scales\ and\ Problem\ Scales --exp_names NeuroGPU-EA CPU-EA CoreNeuronGPU-EA