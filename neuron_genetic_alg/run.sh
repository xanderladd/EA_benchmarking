#!/bin/bash





specimen_id=487664663
storage="/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage"
input_folder="/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_input_files"
preprocess_out="${input_folder}/preprocess_input.json"
cell_folder="/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/cell_${specimen_id}"
surrogate="True"
mu=1200
ngen=500
NUM_PROCS=1200

passive='True'

# 12 + 13 for NaV Ih
# 6 + 9 for regular channels
# fit_type_index_1=12
# fit_type_index_2=13



# make surrogate data
# cd $cell_folder
# nrnivmodl modfiles
# python -m allensdk.model.biophysical.runner manifest.json 
# cp work/* "${specimen_id}_ephys_sim.nwb"
# cd -

#preprocessing
# python -m biophys_optimize.scripts.hack_preprocess_input  $specimen_id $storage $preprocess_out $cell_folder $surrogate

# python -m biophys_optimize.scripts.run_preprocessing --input_json $preprocess_out

# # passive
# passive_1_json="${input_folder}/passive_fit_1.json"
# python -m biophys_optimize.scripts.hack_preprocess_to_passive_strategy  'preprocess_output.json' $passive_1_json passive_fit_1
# python -m biophys_optimize.scripts.run_passive_fitting --input_json $passive_1_json

# passive_2_json="${input_folder}/passive_fit_2.json"
# python -m biophys_optimize.scripts.hack_preprocess_to_passive_strategy  'preprocess_output.json' $passive_2_json passive_fit_2
# python -m biophys_optimize.scripts.run_passive_fitting --input_json $passive_2_json

# passive_elec_json="${input_folder}/passive_fit_elec.json"
# python -m biophys_optimize.scripts.hack_preprocess_to_passive_strategy  'preprocess_output.json' $passive_elec_json passive_fit_elec
# python -m biophys_optimize.scripts.run_passive_fitting --input_json $passive_elec_json

# consolidate_json="${input_folder}/consolidate_input.json"
# python -m biophys_optimize.scripts.hack_consolidate_passive_strategy 'preprocess_output.json' 'passive_fit_1.json' 'passive_fit_2.json' 'passive_fit_elec.json' $consolidate_json
# python -m biophys_optimize.scripts.run_consolidate_passive_fitting --input_json $consolidate_json



module load cray-python/3.7.3.2
module load openmpi
export PYTHONPATH=/global/project/projectdirs/m2043/$USER/install/lib/python:$PYTHONPATH
export PATH=/global/project/projectdirs/m2043/$USER/install/bin:$PATH
export LD_LIBRARY_PATH=/global/project/projectdirs/m2043/$USER/install/lib:$LD_LIBRARY_PATH




# for i in {1..5}
# do
#     optimize_json="${input_folder}/optimize_input.json"
#     salloc --nodes 19 --qos interactive --time 02:40:00 --constraint haswell --no-shell
#     id=`squeue -u zladd -t R -p interactive | awk -F'[^0-9]+' '{ print $2 }'`
#     if [ -z "$id" ]
#     then
#           echo "\$id is empty!! alloc failed!! exiting!!"
#           exit
#     fi
#     python -m biophys_optimize.scripts.hack_optimize_input 'preprocess_output.json' $optimize_json --fit_type_index $i --stage_key stage_1_task_list --mu $mu --ngen $ngen
#     echo "launching stg1 ${i}"
#     srun -n $NUM_PROCS --jobid $id sh optimize.sh $optimize_json

# #     srun -n $NUM_PROCS --jobid $id python -m biophys_optimize.scripts.run_optimize --input_json $optimize_json
#     scancel $id
# done

# f12_model="${input_folder}/f12_model_select_input.json"
# python -m biophys_optimize.scripts.hack_model_select_input 'preprocess_output.json' $f12_model f12

# python -m biophys_optimize.scripts.run_population_selection --input_json $f12_model

# sp=`jq .paths.starting_populations.f12.hof $storage/f12/select.json`
# sp=` echo "$sp" | tr -d '"'`
# echo "sp is ${sp}"


f6_model="${input_folder}/f6_model_select_input.json"
python -m biophys_optimize.scripts.hack_model_select_input 'preprocess_output.json' $f6_model f6

python -m biophys_optimize.scripts.run_population_selection --input_json $f6_model

sp=`jq .paths.starting_populations.f6.hof $storage/f6/select.json`
sp=` echo "$sp" | tr -d '"'`
echo "sp is ${sp}"



for i in {1..5}
do
    optimize_json="${input_folder}/optimize_input.json"
    salloc --nodes 19 --qos interactive --time 02:40:00 --constraint haswell --no-shell
    id=`squeue -u zladd -t R -p interactive | awk -F'[^0-9]+' '{ print $2 }'`
    if [ -z "$id" ]
    then
          echo "\$id is empty!! alloc failed!! exiting!!"
          exit
    fi
    python -m biophys_optimize.scripts.hack_optimize_input 'preprocess_output.json' $optimize_json --fit_type_index $i --stage_key stage_2_task_list --mu $mu --ngen $ngen --sp $sp
    echo "launching stg2 ${i}"
    

    srun -n $NUM_PROCS --jobid $id  sh optimize.sh $optimize_json
    #python -m biophys_optimize.scripts.run_optimize --input_json $optimize_json
    scancel $id
done



f13_model="${input_folder}/f13_model_select_input.json"
python -m biophys_optimize.scripts.hack_model_select_input 'preprocess_output.json' $f13_model f13

python -m biophys_optimize.scripts.run_population_selection --input_json $f13_model










# python -m biophys_optimize.scripts.run_model_selection --input_json $f12_model
# python -m biophys_optimize.scripts.run_population_selection --input_json /global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_input_files/test_population_input.json

