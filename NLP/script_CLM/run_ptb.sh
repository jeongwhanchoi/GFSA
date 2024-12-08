export DEVICE=$1

python -u run_clm_no_trainer.py   \
        --per_device_train_batch_size 4 \
        --dataset_name ptb_text_only \
        --dataset_config_name penn_treebank \
        --model_name_or_path gpt2 \
        --num_train_epochs 15 \
        --mode gfsa \
        --K 7 \
        --seed 2023 \
        --output_dir ./experiments/gpt2_language-modeling/ptb
