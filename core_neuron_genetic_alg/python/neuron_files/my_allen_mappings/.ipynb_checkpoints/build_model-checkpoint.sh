#!/bin/bash


cd /global/cscratch1/sd/zladd/axonstandardized/playground/genetic_alg/make_models_mappings/allen_mappings
python update_params.py $1
echo "updated params so e_pas is $1"

python extractModel.py False
sh moveFiles.sh
cp Sim.csv pyNeuroGPU_unix2/Data/

cd pyNeuroGPU_unix2/python
python extractModel.py True
cp Data/* ../Data
python extractModel_mappings.py
cp ../../Sim.csv ../Data/


srun -n 1 python runAllen.py 


