import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import cv2

import sys
sys.path.append("/data3/xytan/code/") 
#import DropPos.util.misc as misc
from DropPos.util import misc as misc
from DropPos.util.misc import NativeScalerWithGradNormCount as NativeScaler
from DropPos.util.datasets import ImageListFolder
from DropPos.util.pos_embed import interpolate_pos_embed

dataloader_kwargs = {'num_workers': 0, 'pin_memory': True}
def GetDataLoaderDict(dataset_dict, batch_size):
    dataloader_dict = {}
    for dataset_name in dataset_dict.keys():
        if 'train' in dataset_name:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_dict[dataset_name], num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], sampler=sampler_train, batch_size=batch_size, drop_last=True,   **dataloader_kwargs)
        else:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_dict[dataset_name], num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], sampler=sampler_train, batch_size=batch_size, drop_last=False, **dataloader_kwargs)

    return dataloader_dict


class MetaDataset(Dataset):
    '''
    For RGB data, single client
    '''
    def __init__(self, imgs, labels, domain_label, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]
        #print(img_path  + str(img_class_label))
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_class_label, self.domain_label

    def __len__(self):
        return len(self.imgs)
    