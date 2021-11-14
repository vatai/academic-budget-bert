#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=25:00:00
#$ -o /groups/gcb50300/data/NLP/academic-budget-ckpt/$JOB_NAME-$JOB_ID/output.txt
#$ -cwd
#$ -j y

source ./common.src

deepspeed run_pretraining.py \
  --dataset_path $(prev_job_dir sub_samples.sh) \
  --output_dir ${OUTPUT_DIR} \
  --model_type bert-mlm --tokenizer_name bert-base-uncased \
  --hidden_act gelu \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --intermediate_size 3072 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 128 \
  --lr_schedule time \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 24.0 \
  --early_exit_time_marker 24.0 \
  --print_steps 100 \
  --num_epochs_between_checkpoints 1 \
  --job_name "$JOB_NAME" \
  --current_run_id "$JOB_ID" \
  --project_name "$JOB_NAME" \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 180 \
  --early_stop_eval_loss 6 \
  --seed 42 \
  --fp16 \
&& echo All DONE!

  # --load_training_checkpoint /home/acc12262dj/data/data/NLP/academic-training-short-ckpt/pretraining_experiment-/ \
  # --load_training_checkpoint academic-budget-ckpt/sub_base.sh-8290699/8290699 \
deactivate
