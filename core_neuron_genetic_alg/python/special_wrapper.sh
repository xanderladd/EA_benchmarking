#!/bin/bash

#enable mps
nvidia-cuda-mps-control -d > /dev/null 2>&1

# call actual binary
shift
./x86_64/special "$@"