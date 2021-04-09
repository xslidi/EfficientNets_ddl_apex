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
               [--num_workers NUM_WORKERS] [--model MODEL] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE]
               [--test] [--print_freq PRINT_FREQ] [--num_classes NUM_CLASSES]
               [--ema] [--color_jitter COLOR_JITTER] [--pca]
               [--crop_pct CROP_PCT] [--cool_down COOL_DOWN]
               [--ema_decay EMA_DECAY] [--dropout_rate DROPOUT_RATE]
               [--dropconnect_rate DROPCONNECT_RATE]
               [--optim {rmsprop,rmsproptf,sgd}] [--lr LR] [--warmup WARMUP]
               [--beta [BETA [BETA ...]]] [--momentum MOMENTUM] [--eps EPS]
               [--decay DECAY] [--scheduler {exp,cosine,none}] [--amp]
               [--dali] [--se_r SE_R] [--REGNET_WA REGNET_WA]
               [--REGNET_W0 REGNET_W0] [--REGNET_WM REGNET_WM]
               [--REGNET_DEPTH REGNET_DEPTH] [--REGNET_STRIDE REGNET_STRIDE]
               [--REGNET_GROUP_W REGNET_GROUP_W]
               [--REGNET_BOT_MUL REGNET_BOT_MUL]
               [--REGNET_STEM_W REGNET_STEM_W]

Pytorch EfficientNet

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory name to save the model
  --root ROOT           The Directory of data path.
  --gpus GPUS           Select GPU Numbers | 0,1,2,3 |
  --num_workers NUM_WORKERS
                        Select CPU Number workers
  --model MODEL         The type of Efficient net.
  --epoch EPOCH         The number of epochs
  --batch_size BATCH_SIZE
                        The size of batch
  --val_batch_size VAL_BATCH_SIZE
                        The size of batch in val set
  --test                Only Test
  --print_freq PRINT_FREQ
                        The iterations of print results
  --num_classes NUM_CLASSES
                        Number of classes
  --ema                 Using exponential moving average for testing
  --color_jitter COLOR_JITTER
                        Color jitter factor (default: 0.0)
  --pca                 add AlexNet - style PCA - based noise
  --crop_pct CROP_PCT   Input image center crop percent (for validation only)
  --cool_down COOL_DOWN
                        epochs to cooldown LR at min_lr, after cyclic schedule
                        ends
  --ema_decay EMA_DECAY
                        Exponential Moving Average Term
  --dropout_rate DROPOUT_RATE
  --dropconnect_rate DROPCONNECT_RATE
  --optim {rmsprop,rmsproptf,sgd}
  --lr LR               Base learning rate when train batch size is 256.
  --warmup WARMUP
  --beta [BETA [BETA ...]]
  --momentum MOMENTUM
  --eps EPS
  --decay DECAY
  --scheduler {exp,cosine,none}
                        Learning rate scheduler type
  --amp                 Use Native Torch AMP mixed precision
  --dali                Use Naidiv DaLi library for loading
  --se_r SE_R           Squeeze-and-Excitation rate
  --REGNET_WA REGNET_WA
                        Slop
  --REGNET_W0 REGNET_W0
                        Initial width
  --REGNET_WM REGNET_WM
                        Quantization
  --REGNET_DEPTH REGNET_DEPTH
                        Depth
  --REGNET_STRIDE REGNET_STRIDE
                        Stride of each stage
  --REGNET_GROUP_W REGNET_GROUP_W
                        Group width
  --REGNET_BOT_MUL REGNET_BOT_MUL
                        Bottleneck multiplier (bm = 1 / b from the paper)
  --REGNET_STEM_W REGNET_STEM_W
                        Stem width



```

<hr>

### TODO

 - Implementation of Resolution Change
 - Clean up logging

<hr>
