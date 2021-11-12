#!/bin/sh
#$ -l rt_C.small=1
#$ -l h_rt=24:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ../common.src

python3 merge_shards.py \
    --data $(prev_job_dir sub_sharding.sh) \
    --output_dir $(output_dir $JOB_NAME $JOB_ID) \
    --ratio 4 \
&& echo ALL DONE

