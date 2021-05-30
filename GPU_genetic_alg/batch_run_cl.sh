#!/bin/bash



source /gpfs/alpine/scratch/zladd/nro106/axonproj/miniconda3/bin/activate
module unload darshan-runtime/3.1.7

cpus=$1
offspring_size=$2

# # load modules
# module load gcc/7.4.0 
# module load python
# module load cuda/10.1.168 
# module load hdf5
# module load spectrum-mpi/10.3.1.2-2020012

cd /gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/python

#determine number of nodes and total procs
nprocspn=1
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))
echo 'M: nprocs=',$nprocs

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

echo "starting gen_alg with ${cpus}, ${nnodes} nodes and ${offspring_size}" `date`
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} -m 619326  --bind=rs stdbuf -o0  python optimize_parameters_genetic_alg.py --offspring_size $offspring_size --max_ngen 1 

#--checkpoint epas --continu True

#--bind=proportional-packed:7
#--launch_distribution=packed 


setenv USER_POSTEXEC post_exec.sh cpus ${nnodes} offspring_size


