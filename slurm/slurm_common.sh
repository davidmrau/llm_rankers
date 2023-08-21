#!/bin/bash
USER=`accinfo | grep NRC | awk '{print $3}'`
#Loading modules
echo $USER
module purge
module load 2022
#module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
export PYTHONIOENCODING='utf-8'
cd ..
nvidia-smi
FILE_NAME=$1
export FILE_NAME
sleep 2 
source slurm/${1}


