#!/bin/bash
#SBATCH --time=01-00:00                       # time (DD-HH:MM)
#SBATCH --gres=gpu:1                         # Number of GPUs (per node)
#SBATCH --cpus-per-task=1                    # CPU cores/threads
#SBATCH --mem=16G                            # memory per node -> use powers of 2
#SBATCH --qos=normal                         # ICML
#SBATCH --partition=gpu
#SBATCH --output=discussion_gmm.out
#SBATCH --error=discussion_gmm.err

eval "$(conda shell.bash hook)"
conda activate csc401a2

. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/

python -u gmm_discussion.py
