#!/bin/bash

rm -rf Validation_UY.png Validation_XV.png
pylabolt --solver fluidLB --parallel -nt 13 > log.txt 2> err.txt
cp output/50000/fields.dat .
python3 plot.py