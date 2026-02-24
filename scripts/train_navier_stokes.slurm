#!/bin/bash
#SBATCH --job-name=train_ns_default          # 任务名称
#SBATCH --ntasks=1                   # 任务数量
#SBATCH --output=./slurm_output/%x_%j.txt      # 输出文件，%x = job name, %j = job id
#SBATCH --cpus-per-task=10            # 每个任务使用的 CPU 核心数
#SBATCH --partition=GPU              # 选择 GPU 分区
#SBATCH --nodelist=node1 # 选择节点

source ~/.bashrc
conda init bash
conda activate pytorch

srun python train_oformer_ns.py