#!/bin/bash

rm -rf figures/ output/ log* __pycache__/ 
mpirun -np 4 pylabolt --solver fluidLB
mpirun -np 4 pylabolt --reconstruct time -t 72243

python3 plot.py -t 72243 -Nx 101
