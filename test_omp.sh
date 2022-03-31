#!/bin/sh

gcc -o dotprod_opm dotprod_opm.c -fopenmp 
gcc -o matvecmul matvec_mul_omp.c -fopenmp 
gcc -o matvecmulv2 matvec_mul_ompv2.c -fopenmp 
