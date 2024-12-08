# Automatic Speech Recognition
This folder contains the scripts to train a Transformer-based speech recognizer.

You can download LibriSpeech at http://www.openslr.org/12

## Set-up
### Editable install `SpeechBrain`

Clone the repository and install `SpeechBrain` with the following commands:
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install -e .
```

To plug-in GFSA to Transformer and Branchformer, please replace `speechbrain/nnet/attention.py` with `attention.py` in `gfsa` folder.

Replace the original `speechbrain/lobes/models/transformer/{backbone}.py` files with `gfsa/{backbone}.py` files.

The `{backbone}` stands for Transformer and Branchformer.


# How to run
```shell
# pure Transformer
python train.py hparams/transformer.yaml
# pure Transformer + GFSA
python train.py hparams/transformer-gf.yaml
# pure Branchformer
python train.py hparams/branchformer_large.yaml
# pure Branchformer + GFSA
python train.py hparams/branchformer-gf.yaml
```
