export DEVICE=$1

python -u run_clm_no_trainer.py   \
        --per_device_train_batch_size 4 \
        --dataset_name wikitext \
        --dataset_config_name wikitext-103-raw-v1 \
        --model_name_or_path gpt2 \
        --num_train_epochs 4 \
        --mode gfsa \
        --K 3 \
        --seed 2023 \
        --output_dir ./experiments/gpt2_language-modeling/wikitext-2
