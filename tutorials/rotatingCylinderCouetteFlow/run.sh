#!/bin/bash

rm -rf figures/ output/ postProcessing/ log* __pycache__/ 
mpirun -np 9 pylabolt --solver fluidLB
mpirun -np 9 pylabolt --reconstruct all
mpirun -np 9 pylabolt --reconstruct time -t 69600
mpirun -np 9 pylabolt --reconstruct time -t 69641


python3 plot.py -Re 0.1 -Nx 161 -pos 80 -t 69641