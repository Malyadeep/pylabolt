#!/bin/bash

rm -rf figures/ output/ log* __pycache__/ 
pylabolt --solver fluidLB

python3 plot.py -t 50
python3 plot.py -t 500