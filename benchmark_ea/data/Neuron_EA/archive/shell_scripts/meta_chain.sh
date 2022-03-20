#!/bin/bash

echo both scale

rm launch_log.log
touch launch_log.log

jid=`sh weak_chain_jobs.sh plans/Compute_Fixed_and_Problem_Scales ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

# jid=`sh weak_chain_jobs.sh plans/Compute_Scales_and_Problem_Scales_remainder ${jid} | tail -1 | sed 's/[^0-9]*//g'` 


jid=`sh weak_chain_jobs.sh plans/Compute_Fixed_and_Problem_Scales ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

jid=`sh weak_chain_jobs.sh plans/Compute_Fixed_and_Problem_Scales ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

jid=`sh weak_chain_jobs.sh plans/Compute_Fixed_and_Problem_Scales ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

# jid=`sh weak_chain_jobs.sh plans/Compute_Scales_and_Problem_Fixed ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

# jid=`sh weak_chain_jobs.sh plans/Compute_Scales_and_Problem_Scales ${jid} | tail -1 | sed 's/[^0-9]*//g'` 

