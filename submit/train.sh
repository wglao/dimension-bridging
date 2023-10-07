#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
# 
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J dba_train           # Job name
#SBATCH -o dba_train.o%j       # Name of stdout output file
#SBATCH -e dba_train.e%j       # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 4                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH -A DMS21058         # Project/Allocation name (req'd if you have more than 1)

# Any other commands must follow all #SBATCH directives...
pwd
module load launcher
module load tacc-apptainer
module list

export LAUNCHER_WORKDIR=$2
export LAUNCHER_JOB_FILE=$2/argument_files$1

# # Launch serial code...
source ~/.bashrc
${LAUNCHER_DIR}/paramrun
