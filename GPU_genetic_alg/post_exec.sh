#!/bin/bash


FILE=python/runTimeLogs/$1C$2N_$3.log
count=0
while test -f "$FILE"; do
    FILE=python/runTimeLogs/$1C$2N_$3_$count.log
    ((count=count+1))
done

mv python/runTimeLogs/runTime.log $FILE