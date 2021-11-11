#!/bin/sh
#$ -l rt_C.small=1
#$ -l h_rt=24:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-data/wikipedia/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

PREV_JOB_NAME=sub_sharding.sh

source common.src
echo prev_job_dir=$(prev_job_dir $PREV_JOB_NAME)

python3 merge_shards.py \
    --data ${OUTPUT_DIR}/${PREV_JOB_DIR}/output \
    --output_dir ${OUTPUT_DIR}/$JOB_NAME-$JOB_ID/output \
    --ratio 4 \
&& echo ALL DONE

