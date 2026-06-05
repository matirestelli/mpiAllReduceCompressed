#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch        
#SBATCH -J ddp-train             
#SBATCH -N 1                      
#SBATCH --gres=gpu:4                  
#SBATCH -t 01:00:00               
#SBATCH -o ddp_train.%j.out       
#SBATCH -e ddp_train.%j.out       

# srun <your_program>

# Frontier modules
