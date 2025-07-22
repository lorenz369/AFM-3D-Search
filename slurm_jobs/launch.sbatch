#!/bin/bash
#SBATCH --job-name=afm3d-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --output=/usr/prakt/s0118/slurm/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0118@tum.de

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

# Activate your environment if needed
# source /usr/prakt/s0118/miniconda3/bin/activate <your_env_name>

python scripts/eval.py 