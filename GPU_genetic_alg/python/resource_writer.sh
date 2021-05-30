#!/bin/bash

numnodes=$1
numnumnodes=$(expr $numnodes - 1 )

offspringsize=$(expr $numnumnodes '*' 70)

rm -f --  dynamic_resource.txt
touch dynamic_resource.txt

echo "app 0: python optimize_parameters_genetic_alg.py --offspring_size $offspringsize --max_ngen 1" >> dynamic_resource.txt
echo "cpu_index_using: logical" >> dynamic_resource.txt
echo overlapping_rs: allow >> dynamic_resource.txt
echo oversubscribe_cpu: warn >> dynamic_resource.txt
echo oversubscribe_gpu: error >> dynamic_resource.txt
echo oversubscribe_mem: error >> dynamic_resource.txt
#echo launch_distribution: packed >> dynamic_resource.txt
 


for ((i=0; i<=$numnumnodes; i++));
do
    hostnum=$(expr $i + 1)
    echo "rank: $i: { host: $hostnum;  cpu: {0-167} ; gpu: {0,1,2,3,4,5} ; mem: 0-619325 } : app 0" >> dynamic_resource.txt

done

echo finished making resource file w/ $numnodes nodes!!!!

#echo "rank: $i: { host: $hostnum;  cpu: {0-167} ; gpu: {0,1,2,3,4,5} ; mem: {0-409662,409663-619325} } : app 0" >> dynamic_resource.txt


