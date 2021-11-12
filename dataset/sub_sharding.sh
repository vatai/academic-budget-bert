#!/bin/sh
#$ -l rt_C.small=1
#$ -l h_rt=24:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ../common.src

python3 shard_data.py \
  --dir $(prev_job_dir sub_process.sh) \
  -o $(output_dir $JOB_NAME $JOB_ID) \
  --num_train_shards 256 \
  --num_test_shards 128 \
  --frac_test 0.1 \
&& echo ALL DONE

