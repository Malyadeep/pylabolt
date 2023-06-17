#!/bin/bash

rm -rf output/ procs/
mpirun -np 10 pylabolt --solver fluidLB
mpirun -np 10 pylabolt --reconstruct all
pylabolt --toVTK all
