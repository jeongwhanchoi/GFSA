# Code Classification

## Introduction

## Download Pretrained and Fine-tuned Checkpoints

* [Pre-trained checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/pretrained_models)
* [Fine-tuning data](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/data)
* [Fine-tuned checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/finetuned_models)

Instructions to download:

```
# pip install gsutil
cd your-cloned-codet5-path

gsutil -m cp -r "gs://sfr-codet5-data-research/pretrained_models" .
gsutil -m cp -r "gs://sfr-codet5-data-research/data" .
gsutil -m cp -r "gs://sfr-codet5-data-research/finetuned_models" .
```

## Fine-tuning

### Dependency

- Pytorch 1.7.1
- tensorboard 2.4.1
- tree-sitter 0.2.2
- transformers 4.33.0

### Editable install `Transformers`

Clone the repository and install `Transformers` with the following commands:
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```
Replace original `transformers/models/{backbone}/modeling_{backbone}.py` files with our `modeling/modeling_{backbone}.py` files.

The above `{backbone}` stands for Roberta, Bart, CodeBert, and CodeT5.

### How to run?

For example, if you want to run CodeT5-base+GFSA on the code defect prediction task, you can simply run:

```
python run_exp.py --model_tag codet5_base --task defect --sub_task none
```