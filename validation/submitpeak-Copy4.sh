#! /bin/bash
#SBATCH --nodes=4
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -C haswell
#SBATCH -t 00:30:00 ### time needed, debug run 0:30:00
#SBATCH -J offpeakstats5 ## job name
#SBATCH -o offpeakstats5.o%j ## output
#SBATCH -e offpeakstats5.e%j ## error file
#SBATCH --qos=debug #debug #regular #
#SBATCH -A m1727 ## allocation name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgrandons@gmail.com ## change to your email

module load python
source activate my_mpi4py_env
srun python /global/u1/d/dgrandon/code_Peaks-offset-Copy5.py

