#!/bin/bash

rm -rf figures/ output/ log* __pycache__/ 
pylabolt --solver fluidLB --parallel -nt 13

python3 plot.py -t 73278 -Nx 101
