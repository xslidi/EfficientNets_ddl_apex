import torch
import numpy as np
import PIL
import math

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils import ToNumpy, Lighting
from datasets.base import DALIDataloader
from datasets.imagenet import HybridTrainPipe, HybridValPipe
from torchvision.transforms import InterpolationMode

IMAGENET_IMAGES_NUM_TRAIN = 1281166
# IMAGENET_IMAGES_NUM_TRAIN = 50000
IMAGENET_IMAGES_NUM_TEST = 50000
VAL_SIZE = 256
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225) 
IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])}

class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        # if re_prob > 0.:
        #     self.random_erasing = RandomErasing(
        #         probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        # else:
        #     self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                # if self.random_erasing is not None:
                #     next_input = self.random_erasing(next_input)

            if not first:
                yield inputs, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            inputs = next_input
            target = next_target

        yield inputs, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False

def get_loaders(root, batch_size, resolution, num_workers=32, val_batch_size=200, prefetch=False, color_jitter=0.4, pca=False, crop_pct=0.875):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = int(math.floor(resolution / crop_pct))

    transform_train = []
    transform_eval = []

    transform_train += [transforms.RandomResizedCrop(resolution, interpolation=InterpolationMode.BICUBIC),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(*(color_jitter,color_jitter,color_jitter)),]

    transform_eval += [transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
                       transforms.CenterCrop(resolution),]  

    if not prefetch:
        transform_train += [transforms.ToTensor()]
        if pca:
            transform_train += [Lighting(0.1, IMAGENET_PCA['eigval'], IMAGENET_PCA['eigvec'])]            
        transform_train += [normalize,]
        transform_eval += [transforms.ToTensor(),
                            normalize,]
    else:
        transform_train += [ToNumpy()]
        transform_eval += [ToNumpy()]  

    transform_train = transforms.Compose(transform_train)
    transform_eval = transforms.Compose(transform_eval)         

    train_dataset = ImageFolder(
        root + "/train",
        transform_train
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    val_dataset = ImageFolder(
        root + "/val",
        transform_eval
    )

    collate_fn = fast_collate if prefetch else torch.utils.data.dataloader.default_collate
    
    train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler,collate_fn=collate_fn, persistent_workers=True
    )

    val_loader = DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )
    if prefetch:
        train_loader = PrefetchLoader(train_loader)
        val_loader = PrefetchLoader(val_loader)

    return train_loader, val_loader




def get_loaders_dali(root, batch_size, resolution, device_id, world_size, num_workers=32):

    pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers, device_id=device_id, data_dir=root+'/train', crop=resolution, world_size=world_size)

    pip_test = HybridValPipe(batch_size=batch_size, num_threads=num_workers, device_id=device_id, data_dir=root+'/val', crop=resolution, size=VAL_SIZE, world_size=1, local_rank=0)

    size = int(IMAGENET_IMAGES_NUM_TRAIN / world_size)
    
    train_loader = DALIDataloader(pipeline=pip_train, size=size, batch_size=batch_size, onehot_label=True)

    val_loader = DALIDataloader(pipeline=pip_test, size=IMAGENET_IMAGES_NUM_TEST, batch_size=batch_size, onehot_label=True)

    return train_loader, val_loader
