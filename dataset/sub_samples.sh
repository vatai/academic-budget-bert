#!/bin/sh

#$ -l rt_C.large=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -cwd

source /etc/profile.d/modules.sh
# module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
PYENV=pytorch
source ~/${PYENV}.modules
source ~/venv/${PYENV}/bin/activate

python generate_samples.py \
    --dir ~/data/data/NLP/corpora/wikipedia_shards \
    -o ~/data/data/NLP/corpora/wikipedia_samples \
    --dup_factor 10 \
    --seed 42 \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \
    --max_seq_length 128 \
    --vocab_file ~/data/data/NLP/bert-large-uncased-vocab.txt \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 16

