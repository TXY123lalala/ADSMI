import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from utils import ext_transforms as et
import argparse

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

#ls | sed -n 's/\.png$//p' 
# ls | sed -n 's/\.png$//p' ----> mkdir trainVal --> put as train.txt and val.txt
def get_dataset(opts):
    """ Dataset And Augmentation, just for checking!!!
    """
    if opts.dataset == 'BE':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            # et.ExtRandomScale((0.5, 2.0)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            # et.ExtResize( size = 513),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                #                 std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                #                 std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation_polypGen2021(root=opts.data_root,
                                    image_set='train', download=opts.download, transform=train_transform)
        return train_dst


def voc_cmap(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    
    return cmap

class VOCSegmentation_polypGen2021(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 2

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            elem = elem.strip("\n")
            #cut = elem.split(' ')
            #path_current = os.path.join(root, cut[0])
            #target_current = int(cut[1])
            #self.samples.append((path_current, target_current))
            elem = os.path.join('/data3/xytan/data/polypGen_dataset/images_polypGen', elem)+'.jpg'
            self.samples.append((elem,0))
        ann.close()

        print('load finish')



def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
        
def get_argparser():
    parser = argparse.ArgumentParser()
    return parser 

if __name__ == '__main__':
    import torch   
    import matplotlib.pyplot as plt
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    from torch.utils import data
    
    opts = get_argparser()
    opts.data_root = './train_data_polypGen/'
    opts.dataset = 'BE'
    opts.crop_val = False
    opts.crop_size = 512
    opts.download = False
    train_dst = get_dataset(opts) 

    
    train_loader = data.DataLoader(train_dst, batch_size=1, shuffle=True, num_workers=2)    
    print("Train set: %d" % ( len(train_dst)))
    
    for (images, labels) in train_loader:
        print(images.shape)
        print(labels.shape)
        # images = images.to(device, dtype=torch.float32)
        # labels = labels.to(device, dtype=torch.long)
    
    
    # plt.imshow(images[0,:,:,:].permute(1,2,0))
    # plt.imshow(labels[0,:,:])
