#!/bin/sh

$method = 1
$threads = 1
$n = 64000
$m = 64000

while[$threads -le 64]
do
    ./matvec_mul_omp $method $threads $n $m
    wait

    $threads=$(($threads*2))
done