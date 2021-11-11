#!/bin/sh
#$ -l rt_C.small=1
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

python3 process_data.py \
  -f /groups/gcb50300/data/NLP/corpora/original/enwiki-20210101-pages-articles.xml.bz2 \
  -o $OUTPUT_DIR/$JOB_NAME-$JOB_ID/output \
  --type wiki \
&& echo ALL DONE

