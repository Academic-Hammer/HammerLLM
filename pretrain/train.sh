set -ex
export WANDB_PROJECT=hammerllm
BASE_DIR="$PWD"
DATE=$(TZ=Asia/Shanghai date +'%Y%m%d%H%M%S')
CONFIG_PATH=${BASE_DIR}/configs/hammerllm
RUN_NAME=hammerllm_torch_compile_flash_attn_2
OUTPUT_DIR=${BASE_DIR}/checkpoint/${RUN_NAME}

DATA_SEED=3407
MODEL_SEED=3407

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online

if [ ! -d ${OUTPUT_DIR} ]
then
  mkdir -p ${OUTPUT_DIR}
fi
echo "Setting checkpoint directory to ${OUTPUT_DIR}"

MASTER_PORT=$(shuf -n 1 -i 60000-65535)
torchrun --nproc_per_node=8 --master_port ${MASTER_PORT} train.py \
  --model_name_or_path ${CONFIG_PATH} \
  --use_flash_attention_2 \
  --use_torch_compile \
  --train_file /path/to/your/tokenized/train/dataset \
  --validation_files /path/to/your/tokenized/validation/dataset_1 /path/to/your/tokenized/validation/dataset_2 ... \
  --preprocessing_num_workers 100 \
  --block_size 2048 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --max_steps 1000000 \
  --warmup_steps 2000 \
  --eval_steps 500 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --greater_is_better false \
  --load_best_model_at_end false \
  --ddp_find_unused_parameters false \
  --remove_unused_columns false \
  --save_total_limit 50 \
  --learning_rate 5e-4 \
  --lr_scheduler_type cosine \
  --output_dir ${OUTPUT_DIR} \
  --report wandb \
  --run_name ${RUN_NAME} \
  --bf16 \
  --seed ${MODEL_SEED} \
  --data_seed ${DATA_SEED} \
  --deepspeed ${BASE_DIR}/configs/zero_1.json
