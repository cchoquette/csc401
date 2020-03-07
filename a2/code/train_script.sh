#!/bin/bash
#SBATCH --time=00-03:00                       # time (DD-HH:MM) 
#SBATCH --gres=gpu:1                         # Number of GPUs (per node)
#SBATCH --mem=16G                            # memory per node -> use powers of 2 
#SBATCH --cpus-per-task=1                    # CPU cores/threads
#SBATCH --qos=normal                        
#SBATCH --partition=gpu
#SBATCH --output=train_%a.out # specify output file
#SBATCH --error=train_%a.err  # specify error file
#SBATCH --array=1,2

eval "$(conda shell.bash hook)"
conda activate 401a2

TRAIN=/scratch/ssd001/home/cchoquet/csc401/a2/data/Hansard/Training/
TEST=/scratch/ssd001/home/cchoquet/csc401/a2/data/Hansard/Testing/
CELL_TYPE="gru"

is_expid="false"
expid="0"

for arg in $@; do
    if [[ "${arg}" == "--expid" ]]; then
        is_expid="true"
    elif [[ "${is_expid}" == "true" ]]; then
        expid="${arg}"
        is_expid="false"
    fi
done


if [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
	python -u a2_run.py vocab $TRAIN e vocab.e.gz
	python -u a2_run.py vocab $TRAIN f vocab.f.gz
	python -u a2_run.py split $TRAIN train.txt.gz dev.txt.gz
    python -u a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_wo_att.pt.gz --cell-type $CELL_TYPE --device cuda
    python -u a2_run.py test $TEST vocab.e.gz vocab.f.gz model_wo_att.pt.gz --device cuda
elif [[ "${SLURM_ARRAY_TASK_ID}" == "2" ]]; then
	python -u a2_run.py vocab $TRAIN e vocab2.e.gz
	python -u a2_run.py vocab $TRAIN f vocab2.f.gz
    python -u a2_run.py split $TRAIN train2.txt.gz dev2.txt.gz
    python -u a2_run.py train $TRAIN vocab2.e.gz vocab2.f.gz train2.txt.gz dev2.txt.gz model_w_att.pt.gz --cell-type $CELL_TYPE --with-attention --device cuda
    python -u a2_run.py test $TEST vocab2.e.gz vocab2.f.gz model_w_att.pt.gz --cell-type $CELL_TYPE --with-attention --device cuda
fi

