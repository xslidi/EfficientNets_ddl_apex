import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
# from base import DALIDataloader
# from imagenet import HybridTrainPipe, HybridValPipe

IMAGENET_IMAGES_NUM_TRAIN = 1281166
# IMAGENET_IMAGES_NUM_TRAIN = 50000
IMAGENET_IMAGES_NUM_TEST = 50000
VAL_SIZE = 256

def get_loaders(root, batch_size, resolution, num_workers=32, val_batch_size=200):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder(
        root + "/train",
        transforms.Compose([
            transforms.Resize([resolution, resolution]),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    val_dataset = ImageFolder(
        root + "/val",
        transforms.Compose([
            transforms.Resize([resolution, resolution]),
            transforms.ToTensor(),
            normalize,
        ])
    )


    train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )

    val_loader = DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_sampler


# def get_loaders(root, batch_size, resolution, device_id, world_size, num_workers=32):

#     pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers, device_id=device_id, data_dir=root+'/train', crop=resolution, world_size=world_size)

#     pip_test = HybridValPipe(batch_size=batch_size, num_threads=num_workers, device_id=device_id, data_dir=root+'/val', crop=resolution, size=VAL_SIZE, world_size=1, local_rank=0)

#     size = int(IMAGENET_IMAGES_NUM_TRAIN / world_size)
    
#     train_loader = DALIDataloader(pipeline=pip_train, size=size, batch_size=batch_size, onehot_label=True)

#     val_loader = DALIDataloader(pipeline=pip_test, size=IMAGENET_IMAGES_NUM_TEST, batch_size=batch_size, onehot_label=True)

#     return train_loader, val_loader
