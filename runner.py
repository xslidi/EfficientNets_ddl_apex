import os
import copy
import time
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import *

IAGENET_IMAGES_NUM_TEST = 50000


class Runner():
    def __init__(self, arg, net, optim, rank, loss, logger, scheduler=None):
        self.arg = arg
        self.save_dir = arg.save_dir

        self.logger = logger

        self.rank = rank

        self.net = net
        self.ema = copy.deepcopy(net.module).cpu()
        self.ema.eval()
        self.ema_has_module = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema_decay = arg.ema_decay

        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler

        self.start_epoch = 0
        self.best_metric = -1
        if self.rank == 0:
            self.writter = SummaryWriter()

        self.load()


    def save(self, epoch, filename="train"):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """

        torch.save({"model_type": self.arg.model,
                    "start_epoch": epoch + 1,
                    "network": self.net.module.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric,
                    "scheduler": self.scheduler.state_dict()
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

    def load(self, filename=""):
        """ Model load. same with save"""
        if filename == "":
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            ckpoint = torch.load(file_path, map_location=map_location)
            if ckpoint["model_type"] != self.arg.model:
                raise ValueError("Ckpoint Model Type is %s" %
                                 (ckpoint["model_type"]))

            self.net.module.load_state_dict(ckpoint['network'])
            self.ema.load_state_dict(ckpoint['ema'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            self.scheduler.load_state_dict(ckpoint['scheduler'])           
            print("Load Model Type : %s, epoch : %d acc : %f" %
                  (ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def update_ema(self):  
        needs_module = hasattr(self.net, 'module')
        with torch.no_grad():         
            msd = self.net.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach().cpu()
                # if self.rank:
                #     model_v = model_v.to(device=self.rank)
                ema_v.copy_(ema_v * self.ema_decay + (1. - self.ema_decay) * model_v)



    def train(self, train_loader, trainsampler, val_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))        
        self.net.train()

        for epoch in range(self.start_epoch, self.arg.epoch):
            self.net.train()
            trainsampler.set_epoch(epoch)
            
            epoch_start = time.time()
            start_time = time.time()

            if self.arg.amp:
                scaler = torch.cuda.amp.GradScaler()

            for i, (input_, target_) in enumerate(train_loader):
                target_ = target_.to(self.rank, non_blocking=True)
                
                self.optim.zero_grad()
                if self.arg.amp:
                    with torch.cuda.amp.autocast():
                        out = self.net(input_)
                        loss = self.loss(out, target_)

                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()                
                else:
                    out = self.net(input_)
                    loss = self.loss(out, target_)                    
                    loss.backward()                
                    self.optim.step()

                if self.scheduler:
                    self.scheduler.step()
                
                self.update_ema()


                if (i % self.arg.print_freq) == 0:
                    duration = time.time() - start_time
                    if self.rank == 0:
                        lr = self.optim.param_groups[0]['lr']
                        self.logger.log_write("train", epoch=epoch, loss=loss.item(), time=duration, lr=lr)
                        self.writter.add_scalar('train_loss', loss, epoch*len(train_loader.dataset)//self.arg.batch_size + i)
                    start_time = time.time()
                
            if (val_loader is not None) and self.rank == 0:
                self.valid(epoch, val_loader)
            
            epoch_time = time.time() - epoch_start
            print('epoch_time: %.4f' % (epoch_time))
        self.writter.close()


    def _get_acc(self, loader, ema=True):
        # correct = 0
        acc1, acc5, loss = 0, 0, 0
        if not ema:
            self.net.eval()
        with torch.no_grad():
            for input_, target_ in loader:                
                target_ = target_.to(self.rank, non_blocking=True)
                input_ = input_.to(self.rank, non_blocking=True)
                if ema:
                    self.ema.to(self.rank)
                    out = self.ema(input_)
                else:
                    out = self.net(input_)

                loss = self.loss(out, target_)
                # out = F.softmax(out, dim=1)

                # _, idx = out.max(dim=1)
                # correct += (target_ == idx).sum().item()
                _acc1, _acc5 = self.accuracy(out, target_, topk=(1,5))
                acc1 += _acc1[0]
                acc5 += _acc5[0]
                loss += loss
                

        acc1 /= len(loader.dataset)
        acc5 /= len(loader.dataset)
        loss = loss.item() / len(loader.dataset) * self.arg.batch_size
        acc1 = acc1.item()
        acc5 = acc5.item()

        self.ema.to('cpu')
        # return correct / IAGENET_IMAGES_NUM_TEST
        return acc1, acc5, loss

    def valid(self, epoch, val_loader, ema=True):
        start = time.time()
        acc, acc5, loss = self._get_acc(val_loader, ema)
        val_time = time.time() - start
        self.logger.log_write("valid", epoch=epoch, acc=acc, acc5=acc5, loss=loss, time=val_time)
        self.writter.add_scalar('valid_acc_top1', acc, epoch)
        self.writter.add_scalar('valid_acc_top5', acc5, epoch)

        if acc > self.best_metric:
            start = time.time()
            self.best_metric = acc
            self.save(epoch, "epoch[%05d]_acc[%.4f]_top5[%.4f]" % (
                epoch, acc, acc5))
            save_time = time.time() - start
            print('save_time: %.4f s' % (save_time))

    def test(self, train_loader, val_loader):
        print("\n Start Test")
        self.load()
        train_acc = self._get_acc(train_loader)
        valid_acc = self._get_acc(val_loader)
        self.logger.log_write("test", fname="test", train_acc=train_acc, valid_acc=valid_acc)
        return train_acc, valid_acc
    

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            # batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res


