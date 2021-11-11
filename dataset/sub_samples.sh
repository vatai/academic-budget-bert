#!/bin/sh
#$ -l rt_C.large=1
#$ -l h_rt=24:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-data/wikipedia/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source /etc/profile.d/modules.sh
# module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
PYENV=pytorch
source ~/${PYENV}.modules
source ~/venv/${PYENV}/bin/activate

OUTPUT_DIR=/groups/gcb50300/data/NLP/academic-budget-data/wikipedia
PREV_JOB_NAME=sub_merge.sh
PREV_JOB_DIR=$(basename $(ls -d ${OUTPUT_DIR}/${PREV_JOB_NAME}-* | sort | tail -n1))
echo PREV_JOB_DIR=$PREV_JOB_DIR

python3 generate_samples.py \
  --dir /groups/gcb50300/data/NLP/academic-budget-data/wikipedia/sub_merge.sh-$PREV_JOBID/output \
  -o /groups/gcb50300/data/NLP/academic-budget-data/wikipedia/$JOB_NAME-$JOB_ID/output \
  --dup_factor 10 \
  --seed 42 \
  --do_lower_case 1 \
  --masked_lm_prob 0.15 \
  --max_seq_length 128 \
  --vocab_file /groups/gcb50300/data/NLP/academic-budget-data/vocab.txt \
  --model_name bert-base-uncased \
  --max_predictions_per_seq 20 \
  --n_processes $(nproc) \
&& echo All DONE!
