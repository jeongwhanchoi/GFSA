# Image Classification
## Getting Started

### Dependency

Install the following Python libraries which are required to run our code:

```
pytorch 1.7.0
cudatoolkit 11.0
torchvision 0.8.0
timm 0.4.12
```

### Data Preparation

Download and extract ImageNet train and val images from the [official website](http://image-net.org/).
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Usage

### Training

Training GFSA from scratch usually requires multiple GPUs. Please use the following command to train our model with distributed data parallel:

```
python -m torch.distributed.launch --nproc_per_node=<num_nodes> --master_port <port> --use_env \
main.py --auto_reload --model <model_name> --batch-size <batch_size> \
--data-path <data_path> --data-set IMNET --input-size 224 \
--output_dir <log_dir>
```
where `<model_name>` specifies the name of model to build.

To reproduce our results, please follow the command lines below:

<details>
<summary>
12-layer DeiT-S + GFSA
</summary>
```
python -m torch.distributed.launch --nproc_per_node=4 --master_addr=$master_addr --master_port=$MASTER_PORT --use_env \
main.py --auto_reload --model gfsa_small_12 --batch-size 256 \
--data-path /datasets/image --data-set IMNET --input-size 224 \
--output_dir logs/imnet1k_gfsa_small_12_gf
```
</details>

<details>
<summary>
24-layer DeiT-S + GFSA
</summary>
```
python -m torch.distributed.launch --nproc_per_node=4 --master_addr=$master_addr --master_port=$MASTER_PORT --use_env \
main.py --auto_reload --model gfsa_small_24 --batch-size 256 \
--data-path /datasets/image --data-set IMNET --input-size 224 \
--output_dir logs/imnet1k_gfsa_small_24_gf
```
</details>

<details>
<summary>
24-layer CaiT-S + GFSA
</summary>
```
python -m torch.distributed.launch --nproc_per_node=4 --master_addr=$master_addr --master_port=$MASTER_PORT --use_env \
main.py --auto_reload --model gfsa_cait_S24_224 --batch-size 128 \
--data-path /datasets/image --data-set IMNET --input-size 224 --epochs=500 \
--output_dir logs/imnet1k_cait_S24_224_GFSA
```
</details>

