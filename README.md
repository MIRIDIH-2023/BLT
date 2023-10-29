# Bidirectional Layout Transformer (BLT)

This repository contains the code and models for the following paper:

**[BLT: Bidirectional Layout Transformer for Controllable Layout Generation](https://arxiv.org/abs/2112.05112)** In ECCV'22.


If you find this code useful in your research then please cite

```
@article{kong2021blt,
  title={BLT: Bidirectional Layout Transformer for Controllable Layout Generation},
  author={Kong, Xiang and Jiang, Lu and Chang, Huiwen and Zhang, Han and Hao, Yuan and Gong, Haifeng and Essa, Irfan},
  journal={arXiv preprint arXiv:2112.05112},
  year={2021}
}
```

*Please note that this is not an officially supported Google product.*


# Introduction

Automatic generation of such layouts is important as we seek scale-able and diverse visual designs. We introduce BLT, a bidirectional layout transformer. BLT differs from autoregressive decoding as it first generates a draft layout that satisfies the user inputs and then refines the layout iteratively.

## Set up environment

```
conda env create -f environment.yml
conda activate layout_blt
```
and
```
pip install jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Check this out if you meet error when setting up environment

Replaces references to jax.Array with jax.numpy.DeviceArray.

[layout-blt/utils/layout_bert_fast_decode.py commit error](https://github.com/google-research/google-research/commit/89bd283df95962480163778d32ca62baec06392e#diff-d50bc9b308611a6985e4b5a22be2550862a65a951ab4c76909e6318076e9d07e)
[layout-blt/utils/layout_fast_decode.py commit error](https://github.com/google-research/google-research/commit/89bd283df95962480163778d32ca62baec06392e#diff-54e5487c1e5f718c0155f009478d5f506d842f21865583c1a8dd5fdd252314a8)


## Running

```
# Training a model
python  main.py --config configs/${config} --workdir ${model_dir}
# Testing a model
python  main.py --config configs/${config} --workdir ${model_dir} --mode 'test'
```
