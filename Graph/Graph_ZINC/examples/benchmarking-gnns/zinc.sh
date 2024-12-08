#!/usr/bin/env bash

[ -z "${exp_name}" ] && exp_name=zinc
[ -z "${seed}" ] && seed=35
[ -z "${arch}" ] && arch="--ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${warmup_updates}" ] && warmup_updates="40000"
[ -z "${tot_updates}" ] && tot_updates="400000"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "==============================================================================="

CUDA_VISIBLE_DEVICES=0 \
      python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size 256 \
      --dataset_name ZINC \
      --gpus 1 --accelerator ddp --precision 16 \
      $arch \
      --check_val_every_n_epoch 10 --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir ./experiments/zinc \
      --mode gfsa --K 2