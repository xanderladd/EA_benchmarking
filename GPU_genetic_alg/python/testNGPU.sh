#!bin/bash

cd ../src
make clean
make

cd ../python

jsrun  -n 1 -g 6 -c 42 -a 1 --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0  python optimize_parameters_genetic_alg.py --offspring_size 10 --max_ngen 1

