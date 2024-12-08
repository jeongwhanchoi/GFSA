export DEVICE=$1

CUDA_VISIBLE_DEVICES="$DEVICE" python run_glue_no_trainer.py \
  --model_name_or_path "roberta-base" \
  --task_name cola \
  --num_train_epochs 5 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir ./experiments/roberta-base/cola \
  --seed 2023 \
  --learning_rate 2e-5 \
  --mode gfsa \
  --K 6  \
  --w_0 -0.05 \
  --per_device_eval_batch_size 32
