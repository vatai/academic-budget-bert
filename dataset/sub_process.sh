#!/bin/sh
#$ -l rt_C.small=1
#$ -l h_rt=24:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ../common.src

python3 process_data.py \
  -f /groups/gcb50300/data/NLP/corpora/original/enwiki-20210101-pages-articles.xml.bz2 \
  -o $(output_dir $JOB_NAME $JOB_ID) \
  --type wiki \
&& echo ALL DONE

