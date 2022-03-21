#!/bin/bash



currDir=`pwd`

echo "Compiling Neuron"
cd benchmark_ea/data/Neuron_EA/experiments
nrnivmodl neuron_files/allen/
mkdir outputs


# echo "Compiling CoreNeuron"
# cd currDir
# cd benchmark_ea/data/CoreNeuron_EA/experiments
# nrnivmodl -coreneuron coreneuron_files



echo "Compiling NeuroGPU"
cd currDir
cd benchmark_ea/data/NeuroGPU_EA/src/
make
mkdir outputs


echo "Done"
