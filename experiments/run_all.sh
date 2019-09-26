#!/bin/bash

names=( "FUNSD" "data-esposalles" "handwritten_synthetic" )

for i in "${names[@]}"
do 
    ./experiments/run.sh $i
done
