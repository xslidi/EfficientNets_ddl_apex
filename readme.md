# EfficientNet

A PyTorch implementation of `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.`
 on ImageNet

### [[arxiv]](https://arxiv.org/abs/1905.11946) [[Official TF Repo]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)


<hr>

Implement based on Official TF Repo and some other PyTorch implementations. Only opened EfficientNet is included. <br>
This repo not contains baseline network search(Mnas-Net) and compound coefficient search methods.<br>

<b>Some details(HyperParams, transform, EMA ...) are different with Original repo.</b>

## About EfficientNet

If you're new to EfficientNets, here is an explanation straight from the official TensorFlow implementation:

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.


## Pretrained network

This can be trained in imagenet from scratch.

## Retrained performance


Details about the models' performance retrained on ImageNet are below:

|    *Name*         |*# Params*|*Top-1 Acc.*|*Top-5 Acc.*|
|:-----------------:|:--------:|:----------:|:----------:|
| `efficientnet-b0` |   5.3M   |    76.5    |    93.1    |


## How to use:

```
python3 main.py -h
usage: main.py [-h] --save_dir SAVE_DIR [--root ROOT] [--gpus GPUS]
               [--num_workers NUM_WORKERS] [--model {b0}] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE]
               [--test] [--print_freq PRINT_FREQ] [--ema_decay EMA_DECAY]
               [--dropout_rate DROPOUT_RATE]
               [--dropconnect_rate DROPCONNECT_RATE]
               [--optim {rmsprop,rmsproptf}] [--lr LR] [--warmup WARMUP]
               [--beta [BETA [BETA ...]]] [--momentum MOMENTUM] [--eps EPS]
               [--decay DECAY] [--scheduler {exp,cosine,none}] [--amp]


Pytorch EfficientNet

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory name to save the model
  --root ROOT           The Directory of data path.
  --gpus GPUS           Select GPU Numbers | 0,1,2,3 |
  --num_workers NUM_WORKERS
                        Select CPU Number workers
  --model {b0}          The type of Efficient net.
  --epoch EPOCH         The number of epochs
  --batch_size BATCH_SIZE
                        The size of batch
  --val_batch_size VAL_BATCH_SIZE
                        The size of batch in val set
  --test                Only Test
  --print_freq PRINT_FREQ
                        The iterations of print results
  --ema_decay EMA_DECAY
                        Exponential Moving Average Term
  --dropout_rate DROPOUT_RATE
  --dropconnect_rate DROPCONNECT_RATE
  --optim {rmsprop,rmsproptf}
  --lr LR               Base learning rate when train batch size is 256.
  --warmup WARMUP
  --beta [BETA [BETA ...]]
  --momentum MOMENTUM
  --eps EPS
  --decay DECAY
  --scheduler {exp,cosine,none}
                        Learning rate scheduler type
  --amp                 Use Native Torch AMP mixed precision


```

<hr>

### TODO

 - Implementation of Resolution Change
 - Clean up logging

<hr>
