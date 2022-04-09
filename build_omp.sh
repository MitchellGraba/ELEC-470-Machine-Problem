#!/bin/sh

gcc -o dotprod_opm dotprod_opm.c -fopenmp 
gcc -o matvecmul_opm matvec_mul_omp.c -fopenmp 
