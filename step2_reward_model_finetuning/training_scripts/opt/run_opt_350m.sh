#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=2
DATA_PATH="/home/znli/datasets/Dahoas/rm-static"
MODEL_NAME="facebook/opt-350m"
SEED=1234

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step2_reward-${MODEL_NAME/'/'/_}-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT


deepspeed --master_port 12349 main.py \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/opt" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 1 \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
