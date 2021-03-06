import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.optim.lr_scheduler import StepLR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from optim import RMSpropTF
from lr_scheduler import StepLR, CosineAnnealingLR


from models.effnet import EfficientNet
from models.regnet import RegNet
from models.resnet import resnet50
from models.nfnet import NormFreeNet
from runner import Runner
from loader import get_loaders, get_loaders_dali

from logger import Logger

torch.backends.cudnn.benchmark = True

def arg_parse():
    # projects description
    desc = "Pytorch EfficientNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory name to save the model')

    parser.add_argument('--root', type=str, default="/ILSVRC2012_img",
                        help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="3",
                        help="Select GPU Numbers | 0,1,2,3 | ")
    parser.add_argument('--num_workers', type=int, default="4",
                        help="Select CPU Number workers")

    parser.add_argument('--model', type=str, default='regnet_02',
                        help='The type of Efficient net.')
    
    parser.add_argument('--epoch', type=int, default=120, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='The size of batch')
    parser.add_argument('--val_batch_size', type=int, default=200, help='The size of batch in val set')
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--print_freq', type=int, default=50, help='The iterations of print results')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes') 
    parser.add_argument('--ema', action="store_true", help='Using exponential moving average for testing')
    parser.add_argument('--color_jitter', type=float, default=0.0, help='Color jitter factor (default: 0.0)')     
    parser.add_argument('--pca', action="store_true", help='add AlexNet - style PCA - based noise') 
    parser.add_argument('--crop_pct', default=0.875, type=float, help='Input image center crop percent (for validation only)') 
    parser.add_argument('--cool_down', type=int, default=0, help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--no_filter_bias', action="store_true", help='not filter out bias, bn and other 1d params from weight decay') 

    # EfficientNet options
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help="Exponential Moving Average Term")    
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--dropconnect_rate', type=float, default=0.2)

    parser.add_argument('--optim', type=str, default='sgd', choices=["rmsprop","rmsproptf", "sgd"])
    parser.add_argument('--lr',    type=float, default=0.2, help="Base learning rate when train batch size is 256.")
    parser.add_argument('--warmup', type=int, default=5)
    # Optimizer options
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps',      type=float, default=0.001)
    parser.add_argument('--decay',    type=float, default=5e-5)

    parser.add_argument('--scheduler', type=str, default='cosine', choices=["exp", "cosine", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument('--amp', action="store_true", help='Use Native Torch AMP mixed precision')                        
    parser.add_argument('--dali', action="store_true", help='Use Naidiv DaLi library for loading')
    parser.add_argument('--profiler', action="store_true", help='Use profiler tool')

    # RegNet operations
    parser.add_argument('--se_r', type=float, default=0.25, help='Squeeze-and-Excitation rate')
    parser.add_argument('--REGNET_WA', type=float, default=36.44, help='Slop')
    parser.add_argument('--REGNET_W0', type=int, default=24, help='Initial width') 
    parser.add_argument('--REGNET_WM', type=float, default=2.49, help='Quantization') 
    parser.add_argument('--REGNET_DEPTH', type=int, default=13, help='Depth') 
    parser.add_argument('--REGNET_STRIDE', type=int, default=2, help='Stride of each stage')
    parser.add_argument('--REGNET_GROUP_W', type=int, default=8, help='Group width') 
    parser.add_argument('--REGNET_BOT_MUL', type=float, default=1.0, help='Bottleneck multiplier (bm = 1 / b from the paper)')
    parser.add_argument('--REGNET_STEM_W', type=int, default=32, help='Stem width') 

    return parser.parse_args()

def overrider(arg, mfg):
    for k, v in mfg.items():
        vars(arg)[k] = v
    return arg

def get_model(arg, classes=1000):
    if arg.model == "b0":
        return EfficientNet(1, 1, num_classes=classes, drop_connect_rate=arg.dropconnect_rate, dropout_rate=arg.dropout_rate), 224
    elif arg.model == "b1":
        return EfficientNet(1, 1.1, num_classes=classes, drop_connect_rate=arg.dropconnect_rate, dropout_rate=arg.dropout_rate), 240
    elif arg.model == "regnet_02":
        mfg = dict(REGNET_W0=24, REGNET_WA=36.44, REGNET_WM=2.49, REGNET_GROUP_W=8, REGNET_DEPTH=13)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_04":
        mfg = dict(REGNET_W0=48, REGNET_WA=27.89, REGNET_WM=2.09, REGNET_GROUP_W=8, REGNET_DEPTH=16)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_06":
        mfg = dict(REGNET_W0=48, REGNET_WA=32.54, REGNET_WM=2.32, REGNET_GROUP_W=8, REGNET_DEPTH=15)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_08":
        mfg = dict(REGNET_W0=56, REGNET_WA=38.84, REGNET_WM=2.4, REGNET_GROUP_W=16, REGNET_DEPTH=14)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_16":
        mfg = dict(REGNET_W0=48, REGNET_WA=20.71, REGNET_WM=2.65, REGNET_GROUP_W=24, REGNET_DEPTH=27)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_32":
        mfg = dict(REGNET_W0=80, REGNET_WA=42.63, REGNET_WM=2.66, REGNET_GROUP_W=24, REGNET_DEPTH=21)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "regnet_40":
        mfg = dict(REGNET_W0=96, REGNET_WA=31.41, REGNET_WM=2.24, REGNET_GROUP_W=64, REGNET_DEPTH=22)
        arg = overrider(arg, mfg)
        return RegNet(arg), 224
    elif arg.model == "resnet50":
        return resnet50(), 224
    elif arg.model == "resnet50d":
        return resnet50(stem_width=32, stem_type='deep', avg_down=True), 224
    elif arg.model == "seresnet50":
        return resnet50(block_args=dict(attn_layer='se')), 224
    elif arg.model == "ecanet50":
        return resnet50(block_args=dict(attn_layer='eca')), 224
    elif arg.model == "nfregnet_b0":
        return NormFreeNet(arg), 192
    elif arg.model == "nfregnet_b1":
        return NormFreeNet(arg, depths=(2, 4, 7, 7)), 240


def get_scheduler(optim, sche_type, step_size, t_max, warmup_t=0, t_min=0, warmup_lr_init=0):
    if sche_type == "exp":
        # return StepLR(optim, step_size, 0.97)
        return StepLR(optim, step_size, gamma=0.97, warmup_t=warmup_t)
    elif sche_type == "cosine":
        return CosineAnnealingLR(optim, t_max, eta_min=t_min, warmup_t=warmup_t, step_size=step_size, warmup_lr_init=warmup_lr_init)
    else:
        return None

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Filter out bias, bn and other 1d params from weight decay
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, arg):

    logger = Logger(arg.save_dir)
       
    setup(rank, world_size)
    print(rank)

    scaled_lr = arg.lr * arg.batch_size / 256
    arg.batch_size = int(arg.batch_size / world_size)
    num_workers = int(arg.num_workers / world_size)

    net, res = get_model(arg, classes=arg.num_classes) 
    logger.will_write(str(arg) + "\n")
    net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    if not arg.dali:
        train_loader, val_loader = get_loaders(arg.root, arg.batch_size, res, num_workers, arg.val_batch_size, color_jitter=arg.color_jitter, pca=arg.pca, crop_pct=arg.crop_pct)
    else:
        train_loader, val_loader = get_loaders_dali(arg.root, arg.batch_size, res, rank, world_size, num_workers)
    
    # net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss().cuda()

    if not arg.no_filter_bias:
        parameters = add_weight_decay(net, weight_decay=arg.decay)
        weight_decay = 0
        print('filter out bias, bn and other 1d params from weight decay')
    else:
        parameters = net.parameters()
        weight_decay = arg.decay
    
    optim = {
        # "adam" : lambda : torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "sgd": lambda : torch.optim.SGD(parameters, lr=scaled_lr, momentum=arg.momentum, nesterov=True, weight_decay=weight_decay),
        "rmsproptf": lambda : RMSpropTF(parameters, lr=scaled_lr, momentum=arg.momentum, eps=arg.eps, weight_decay=weight_decay),
        "rmsprop" : lambda : torch.optim.RMSprop(parameters, lr=scaled_lr, momentum=arg.momentum, eps=arg.eps, weight_decay=weight_decay)
    }[arg.optim]()

    scheduler = get_scheduler(optim, arg.scheduler, int(1.0 * len(train_loader)), arg.epoch * len(train_loader), warmup_t=int(arg.warmup * len(train_loader)), warmup_lr_init=0.1 * scaled_lr)

    arg.epoch = arg.epoch + arg.cool_down if arg.cool_down > 0 else arg.epoch
    model = Runner(arg, net, optim, rank, loss, logger, scheduler, world_size)

    if arg.profiler:
        model.profiler(train_loader, val_loader, train_loader.sampler)

    if arg.test is False:
        model.train(train_loader, val_loader)
        cleanup()
    # model.test(train_loader, val_loader, arg.ema)
    
def run_wrap(main_worker, world_size, arg):

    mp.spawn(main_worker,
             args=(world_size, arg, ),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    arg = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus

    arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    mp.set_start_method("spawn")
    run_wrap(main, n_gpus, arg)
