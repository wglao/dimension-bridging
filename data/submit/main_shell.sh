#!/bin/bash
# Information about the parameters
# $1: number of nodes
# $2: number of tasks per nodes: Frontera: 56 

rm *.e*
rm *.o*
rm argument_files*

python generate_argument_files.py --nodes $1 --tasks $2
for param in $(seq 1 1 $1)
do
	sbatch ./postproc.sh $param $PWD
done


