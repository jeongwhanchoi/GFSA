# Natural Language Understanding
Our code is based on the codebase of [transformers](https://github.com/huggingface/transformers) produced by hugging-face with version 4.32.0.dev0.
For plugging GFSA in BERT, we modified [modeling_bert.py](./transformers/models/bert/modeling_bert.py) and [configuration_bert.py](./transformers/models/bert/configuration_bert.py) for plugging GFSA in BERT.

## Environment
Create conda environment with [environment_gfsa_nlu.yaml](./environment_gfsa_nlu.yaml). All the experiments of NLU in our paper are conducted on 1 GPU of NVIDIA A5000 24GB.

## Examples
### BERT finetuned on CoLA
```
sh script_NLU/run_bert.sh 0 
```
### ALBERT finetuned on CoLA
```
sh script_NLU/run_albert.sh 0 
```
### RoBERTa finetuned on CoLA
```
sh script_NLU/run_roberta.sh 0 
```

# Causal Language Modeling
Our code is based on the codebase of [transformers](https://github.com/huggingface/transformers) produced by hugging-face with version 4.32.0.dev0. For plugging GFSA in GPT2, we modified [modeling_gpt2.py](./transformers/models/gpt2/modeling_gpt2.py) and [configuration_gpt2.py](./transformers/models/gpt2/configuration_gpt2.py) for plugging GFSA in GPT2.

## Environment
Create conda environment with [environment_gfsa_clm.yaml](./environment_gfsa_clm.yaml). All the experiments of CLM in our paper are conducted on 1 GPU of NVIDIA RTX 3090 24GB.


## Examples
### GPT2 finetuned on PTB
```
sh script_CLM/run_ptb.sh 0 
```
### GPT2 finetuned on WikiText-2
```
sh script_CLM/run_wikitext2.sh 0 
```
### GPT2 finetuned on WikiText-103
```
sh script_CLM/run_wikitet103.sh 0 
```