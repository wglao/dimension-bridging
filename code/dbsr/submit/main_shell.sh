#!/bin/bash
# Information about the parameters
# $1: main file: main.py
# $2: number of nodes
# $3: number of tasks per nodes: Frontera: 56 

rm *sr*.e*
rm *sr*.o*
rm argument_files*

python generate_argument_files.py --file $1 --nodes $2 --tasks $3
for param in $(seq 1 1 $2)
do
	sbatch ./train.sh $param $PWD
done


