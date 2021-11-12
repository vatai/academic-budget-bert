#!/bin/sh
#$ -l rt_C.large=1
#$ -l h_rt=34:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ../common.src

python3 generate_samples.py \
  --dir $(prev_job_dir sub_merge.sh) \
  -o $(output_dir $JOB_NAME $JOB_ID) \
  --dup_factor 10 \
  --seed 42 \
  --do_lower_case 1 \
  --masked_lm_prob 0.15 \
  --max_seq_length 128 \
  --vocab_file /groups/gcb50300/data/NLP/academic-budget-ckpt/vocab.txt \
  --model_name bert-base-uncased \
  --max_predictions_per_seq 20 \
  --n_processes $(nproc) \
&& echo All DONE!
