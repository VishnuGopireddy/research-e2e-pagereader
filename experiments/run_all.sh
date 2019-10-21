#!/bin/bash
./experiments/set_path.sh

names=( "data-esposalles" "handwritten_synthetic" "FUNSD" )

./experiments/run_its.sh 

for i in "${names[@]}"
do 
    ./experiments/run.sh $i
done
