#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=15
#SBATCH --cpus-per-task=4
#SBATCH -t 128:00:00 
#SBATCH --job-name=KFInSAR 
#SBATCH --mem=0 
##SBATCH --partition=XXXXX 
##SBATCH --mail-type=ALL
##SBATCH --mail-user=YOUREMAIL

source ~/.bashrc
export OMP_NUM_THREADS=8

mpirun -n 30 python -u kfts.py -c configs/refconfigfile.ini

echo Time is `date`
