# EfficientNet

A PyTorch implementation of `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.`


### [[arxiv]](https://arxiv.org/abs/1905.11946) [[Official TF Repo]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)


<hr>

Implement based on Official TF Repo. Only opened EfficientNet is included. <br>
This repo not contains baseline network search(Mnas-Net) and compound coefficient search methods.<br>

<b>Some details(HyperParams, transform, EMA ...) are different with Original repo.</b>


## Pretrained network

This can be trained in imagenet from scratch.



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
               [--decay DECAY] [--scheduler {exp,cosine,none}]


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


```

<hr>

### TODO

 - Hyper Parameter / Imagenet Transformation Check
 - Implementation of Resolution Change
 - Clean up logging

<hr>
