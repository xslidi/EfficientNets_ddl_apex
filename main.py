import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim import RMSpropTF
from lr_scheduler import StepLR


from models.effnet import EfficientNet
from runner import Runner
from loader import get_loaders, get_loaders_dali


from logger import Logger


def arg_parse():
    # projects description
    desc = "Pytorch EfficientNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory name to save the model')

    parser.add_argument('--root', type=str, default="/ILSVRC2012_img",
                        help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="0",
                        help="Select GPU Numbers | 0,1,2,3 | ")
    parser.add_argument('--num_workers', type=int, default="4",
                        help="Select CPU Number workers")

    parser.add_argument('--model', type=str, default='b0',
                        choices=["b0", "b1"],
                        help='The type of Efficient net.')
    
    parser.add_argument('--epoch', type=int, default=350, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch')
    parser.add_argument('--val_batch_size', type=int, default=256, help='The size of batch in val set')
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--print_freq', type=int, default=50, help='The iterations of print results')

    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help="Exponential Moving Average Term")

    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--dropconnect_rate', type=float, default=0.2)

    parser.add_argument('--optim', type=str, default='rmsproptf', choices=["rmsprop","rmsproptf"])
    parser.add_argument('--lr',    type=float, default=0.016, help="Base learning rate when train batch size is 256.")
    parser.add_argument('--warmup', type=int, default=5)
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps',      type=float, default=0.001)
    parser.add_argument('--decay',    type=float, default=1e-5)

    parser.add_argument('--scheduler', type=str, default='exp', choices=["exp", "cosine", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument('--amp', action="store_true", help='Use Native Torch AMP mixed precision')                        
    parser.add_argument('--dali', action="store_true", help='Use Naidiv DaLi library for loading') 
    
    return parser.parse_args()


def get_model(arg, classes=1000):
    if arg.model == "b0":
        return EfficientNet(1, 1, num_classes=classes), 224
    elif arg.model == "b1":
        return EfficientNet(1, 1.1, num_classes=classes), 240


def get_scheduler(optim, sche_type, step_size, t_max, warmup_t=0):
    if sche_type == "exp":
        # return StepLR(optim, step_size, 0.97)
        return StepLR(optim, step_size, gamma=0.97, warmup_t=warmup_t)
    elif sche_type == "cosine":
        return CosineAnnealingLR(optim, t_max)
    else:
        return None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, arg):

    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")
       
    setup(rank, world_size)
    print(rank)

    scaled_lr = arg.lr * arg.batch_size / 256
    arg.batch_size = int(arg.batch_size / world_size)
    arg.num_workers = int(arg.num_workers / world_size)

    net, res = get_model(arg, classes=1000) 
    net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    if arg.dali:
        train_loader, val_loader, train_sampler = get_loaders(arg.root, arg.batch_size, res, arg.num_workers, arg.val_batch_size)
    else:
        train_loader, val_loader = get_loaders_dali(arg.root, arg.batch_size, res, rank, world_size, arg.num_workers)
    
    # net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()
    
    optim = {
        # "adam" : lambda : torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "rmsproptf": lambda : RMSpropTF(net.parameters(), lr=scaled_lr, momentum=arg.momentum, eps=arg.eps, weight_decay=arg.decay),
        "rmsprop" : lambda : torch.optim.RMSprop(net.parameters(), lr=scaled_lr, momentum=arg.momentum, eps=arg.eps, weight_decay=arg.decay)
    }[arg.optim]()

    scheduler = get_scheduler(optim, arg.scheduler, int(2.4 * len(train_loader)), arg.epoch * len(train_loader), warmup_t=int(arg.warmup * len(train_loader)))

    model = Runner(arg, net, optim, rank, loss, logger, scheduler)
    if arg.test is False:
        if arg.dali:
            model.train(train_loader, train_sampler, val_loader)
        else:
            model.train(train_loader, val_loader)
        cleanup()
    model.test(train_loader, val_loader)
    
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
    run_wrap(main, n_gpus, arg)
