#!/bin/sh

gcc -o dotprod_omp dotprod_opm.c -fopenmp 
gcc -o matvecmul_omp matvec_mul_omp.c -fopenmp 
