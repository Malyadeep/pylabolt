#!/bin/bash

rm -rf uLine.png fields_incompressible.dat fields_compressible.dat
pylabolt --solver fluidLB --parallel -nt 13 > log_incompressible.txt 2> err_compressible.txt
mv output/100000/fields.dat ./fields_incompressible.dat

sed -i "s/incompressible/secondOrder/g" simulation.py
pylabolt --solver fluidLB --parallel -nt 13 > log_compressible.txt 2> log_incompressible.txt
mv output/100000/fields.dat ./fields_compressible.dat

python3 plot.py