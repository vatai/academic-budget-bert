#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=25:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ./common.src

python3 run_glue.py \
  --model_name_or_path ${OUTPUT_DIR}/training-large-ckpt/pretraining_experiment-/epoch1000000_step14823/ \
  --output_dir $(output_dir $JOB_NAME $JOB_ID) \
  --task_name MNLI \
  --max_seq_length 128 \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 1 --evaluation_strategy epoch \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 \
&& echo All DONE!
