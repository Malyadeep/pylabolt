#!/bin/bash

rm -rf uLine.png fields_incompressible.dat fields_compressible.dat
mpirun -np 6 pylabolt --solver fluidLB > log_incompressible.txt 2> err_incompressible.txt
mpirun -np 6 pylabolt --reconstruct last
mv output/300000/fields.dat ./fields_incompressible.dat

sed -i "s/incompressible/secondOrder/g" simulation.py
mpirun -np 6 pylabolt --solver fluidLB > log_compressible.txt 2> log_incompressible.txt
mpirun -np 6 pylabolt --reconstruct last
mv output/300000/fields.dat ./fields_compressible.dat

python3 plot.py