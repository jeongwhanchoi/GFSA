# Swin + GFSA

## Training

Training GFSA from scratch usually requires multiple GPUs. Please use the following command to train our model with distributed data parallel:

To reproduce our results, please follow the command lines below:

```
python -m torch.distributed.launch --nproc_per_node=$gpu --master_addr=$master_addr --master_port=$MASTER_PORT \
main.py --cfg configs/attngf_small_patch4_window7_224.yaml --batch-size 128 \
--data-path /datasets/image --data-set IMNET --input-size 224 \
--output output --tag gfsa
```