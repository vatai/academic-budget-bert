#!/bin/sh

#$ -l rt_C.large=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/pytorch+horovod/bin/activate

python3 process_data.py -f ~/data/data/NLP/corpora/original/enwiki-20210101-pages-articles.xml.bz2 -o ~/data/data/NLP/corpora/wikipedia_proc_for_budget_paper --type wiki

