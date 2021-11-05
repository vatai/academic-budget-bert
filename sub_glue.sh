#!/bin/sh

#$ -l rt_F=1
#$ -j y
#$ -l h_rt=2:30:00
#$ -cwd
#$ -o ~/shared/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/pytorch+horovod/bin/activate

NUM_GPUS_PER_NODE=4
NUM_PROCS=$(expr ${NHOSTS} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0"

python3 run_glue.py \
  --model_name_or_path ~/shared/data/NLP/academic-budget-ckpt \
  --task_name MNLI \
  --max_seq_length 128 \
  --output_dir ~/data/data/NLP/academic-training-short-ckpt-ft \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 50 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50
