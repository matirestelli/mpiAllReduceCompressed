#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J ddp-train             
#SBATCH -N 1                      
#SBATCH --gres=gpu:4                  
#SBATCH -t 01:00:00               
#SBATCH -o ddp_train.%j.out       
#SBATCH -e ddp_train.%j.out       

# srun <your_program>
# Frontier modules

source envScriptFrontier.sh


#ask for a compute node 
# salloc -A gen243 -p batch -J installingPyTorch -N 1 --gres=gpu:4 -t 01:00:00 -o installingPyTorch.%j.out -e installingPyTorch.%j.out

# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00
# NB to use nvme you need to request that with the allocation, if sbatch sh job launche out there if interactive terminal :
# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00 -C nvme

# salloc -A gen243 -p batch -J torch-build-and-pack -N 1 -t 00:30:00 -c 16
