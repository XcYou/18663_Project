#!/bin/bash

LINES=3

#nvcc ../cuda_matrix_global.cu -o ./cg.o -D __NO_OUTPUT -D __TIME -Wno-deprecated-gpu-targets
nvcc -arch=sm_35 -rdc=true ../cuda_conv_binary.cu -o ./cs.o -D __NO_OUTPUT -D __TIME -Wno-deprecated-gpu-targets
#nvcc ../cuda_matrix_shared.cu -o ./cs.o -D __NO_OUTPUT -D __TIME -Wno-deprecated-gpu-targets
awk 'BEGIN { printf "%-7s %-20s\n", "LINES", "TIME_SHARED"}'
TIME_SHARED_TOTAL=0
COUNTER=0

python3 ../generate_input.py $LINES $LINES $LINES
./cs.o < input
