#!/bin/bash
# Information about the parameters
# $1: main file: main.py
# $2: number of slices
# $3: number of nodes
# $4: number of tasks per nodes: Frontera: 4 

rm *.e*
rm *.o*
rm process_argument_files*

python generate_process_argument_files.py --file $1 --slices $2 --nodes $3 --tasks $4
for param in $(seq 1 1 $3)
do
	sbatch ./process_dataset.sh $param $PWD
done


