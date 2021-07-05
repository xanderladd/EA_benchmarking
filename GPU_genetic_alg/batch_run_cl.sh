#!/bin/bash



cpus=$1
offspring_size=$2
n_stim=$3
n_sf=$4



# eval source /gpfs/alpine/scratch/zladd/nro106/axonproj/miniconda3/bin/activate
# conda activate /gpfs/alpine/scratch/zladd/nro106/axonproj/miniconda3/envs/dot_env
# module unload darshan-runtime/3.1.7module unload darshan-runtime/3.1.7

#source ./activate_env.sh

# export USER_POSTEXEC="sh /gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/post_exec.sh $cpus ${nnodes} $offspring_size"

# export POST_EXEC="sh /gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/post_exec.sh $cpus ${nnodes} $offspring_size"


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

post_exec_ran=0

post_exec()
{
    nickname=${nnodes}N_${cpus}C_${offspring_size}O_${n_stim}S_${n_sf}SF
    count=0
    FILE=runTimeLogs/${nickname}_${count}.log
    nickname_ct=${nickname}_${count}
    while test -f "$FILE"; do
        FILE=runTimeLogs/${nickname}_${count}.log
        nickname_ct=${nickname}_${count}
        ((count=count+1))
    done

    mv runTimeLogs/runTime.log $FILE
    mkdir outputs/$nickname_ct
    mv gpu_utillization.log outputs/${nickname_ct}/gpu_utillization.log
    cp $FILE outputs/${nickname_ct}/
    #cp GPUTilLogs/${cpus}C${nnodes}N_80_$count_utilization.log outputs/${cpus}C${nnodes}N_80_$count/
    mkdir outputs/${nickname_ct}/profiles
    mv profiles/* outputs/${nickname_ct}/profiles
    mv *.prof outputs/${nickname_ct}/ # could be more specific but we should only have one prof here
    mv ../logs/* outputs/${nickname_ct}/
    post_exec_ran=1
    echo `pwd`
    echo "------- DONE ----------"
}

# call your_cleanup_function once we receive USR1 signal
trap 'post_exec' USR1 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

echo "starting gen_alg with ${cpus}, ${nnodes} nodes and ${offspring_size}" `date`
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} -m 619310  --bind=rs stdbuf -o0  python -m cProfile -o ${nnodes}N${cpus}C${offspring_size}O.prof optimize_parameters_genetic_alg.py --offspring_size $offspring_size --max_ngen 1 --n_stims $n_stim --n_sfs $n_sf & wait

if [ $post_exec_ran -eq 0 ]; then
    post_exec
fi

#--checkpoint epas --continu True

#--bind=proportional-packed:7
#--launch_distribution=packed 


# sh /gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/post_exec.sh $cpus ${nnodes} $offspring_size




