#!/bin/sh

gcc -o dotprod_opm dotprod_opm.c -fopenmp 

./dotprod_opm