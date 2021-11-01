#!/bin/sh
#$ -cwd
#$ -j y
#$ -l rt_C.large=1
#$ -l h_rt=24:00:00


source /etc/profile.d/modules.sh
# module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ${HOME}/pytorch.modules
source ${HOME}/venv/pytorch/bin/activate

INDIR=/groups/gcb50300/data/NLP/corpora/wikipedia_proc_for_budget_paper
OUTDIR=/groups/gcb50300/data/NLP/corpora/wikipedia_shards

python3 shard_data.py \
  --dir $INDIR -o $OUTDIR \
  --num_train_shards 256 \
  --num_test_shards 128 \
  --frac_test 0.1

