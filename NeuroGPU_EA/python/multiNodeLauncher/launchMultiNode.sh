#! /bin/bash

# first job - no dependencies
jid1=$(sbatch   job1.sh) # 2 nodes x 8 gpus

# multiple jobs can depend on a single job
slurm1=$(echo $jid1 | sed 's/[^0-9]*//g')
jid2=$(sbatch  --dependency=afterany:$slurm1  job2.sh)  # 3 nodes x 6 gpus
slurm2=$(echo $jid2 | sed 's/[^0-9]*//g')
jid3=$(sbatch  --dependency=afterany:$slurm2  job3.sh) # 4 nodes x 5 gpus
slurm3=$(echo $jid3 | sed 's/[^0-9]*//g')
jid4=$(sbatch  --dependency=afterany:$slurm3  job4.sh) # 5 nodes x 4 gpus
slurm4=$(echo $jid4 | sed 's/[^0-9]*//g')
jid5=$(sbatch  --dependency=afterany:$slurm4  job5.sh) # 6 nodes x 3 gpus
slurm5=$(echo $jid5 | sed 's/[^0-9]*//g')
jid6=$(sbatch  --dependency=afterany:$slurm5  job6.sh)  # 9 nodes x 2 gpus
slurm6=$(echo $jid6 | sed 's/[^0-9]*//g')
jid7=$(sbatch  --dependency=afterany:$slurm6  job7.sh) # 8 nodes x 3 gpus
slurm7=$(echo $jid7 | sed 's/[^0-9]*//g')
jid8=$(sbatch  --dependency=afterany:$slurm7  job8.sh) # 9 nodes x 2 gpus

