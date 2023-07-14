#!/bin/bash

rm -rf uLine.png fields_incompressible.dat fields_compressible.dat
mpirun -np 3 pylabolt --solver fluidLB > log_incompressible.txt 2> err_incompressible.txt
mpirun -np 3 pylabolt --reconstruct time -t 7251
mv output/7251/fields.dat ./fields_incompressible.dat

sed -i "s/incompressible/secondOrder/g" simulation.py
mpirun -np 3 pylabolt --solver fluidLB > log_compressible.txt 2> err_compressible.txt
mpirun -np 3 pylabolt --reconstruct time -t 17013
mv output/17013/fields.dat ./fields_compressible.dat

python3 plot.py
