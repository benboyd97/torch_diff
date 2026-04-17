#!/bin/bash
#SBATCH --time=00:29:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --account=m1727_g
#SBATCH --job-name=symo_full
#SBATCH -c 8
#SBATCH -q debug
#SBATCH --mail-type=ALL

module load cudatoolkit cudnn
module load python
conda activate /pscratch/sd/b/benboyd/envs/symo_torch311
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u $PSCRATCH/torch_diff/train_script.py

