import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from configs.default import pacs_path
from torchvision import transforms
from utils import ext_transforms as et
#from data import VOCSegmentation_polypGen2021 as polyGenSeg
from data.polypGen2021 import VOCSegmentation_polypGen2021 as polyGenSeg

import sys
sys.path.append("/data/tyc/code/txy") 
#import DropPos.util.misc as misc
from DropPos.util import misc as misc
from DropPos.util.misc import NativeScalerWithGradNormCount as NativeScaler
from DropPos.util.datasets import ImageListFolder
from DropPos.util.pos_embed import interpolate_pos_embed
from .mask_transform import MaskTransform
from ADA_data.concat_dataset import ConcatDataset

"""
pacs_name_dict = {
    'Colonoscopic': 'Colonoscopic',
    'CVC-ClinicDB': 'CVC-ClinicDB',
    'LDPolypVideo': 'LDPolypVideo',
}
"""


pacs_name_dict = {
    'C_1': 'C_1',
    'C_2': 'C_2',
    'C_3': 'C_3',
    'C_4': 'C_4',
    'C_5': 'C_5',
    'C_6': 'C_6',
}

split_dict = {
    'train': 'train',
    #'val': 'crossval',
    'val': 'test',
    'total': 'test',
}

class PACS_SingleDomain():
    def __init__(self, args, root_path=pacs_path, domain_name='C_1', split='total'):
        if domain_name in pacs_name_dict.keys():
            self.domain_name = pacs_name_dict[domain_name]
            self.domain_label = list(pacs_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in p a c s')
        
        #self.root_path = os.path.join(root_path, 'raw_images')
        self.root_path = root_path
        self.split = split
        #self.split_file = os.path.join(root_path, 'raw_images', 'splits', f'{self.domain_name}_{split_dict[self.split]}' + '.txt')
        path = os.path.join(self.root_path, domain_name)
        ann_file = os.path.join(path, split + '.txt')
        
        transform_train = MaskTransform(args)
        dataset_temp = polyGenSeg(root=path, transform=transform_train, ann_file = ann_file)
        #self.dataset = polyGenSeg(root=path, transform=transform_train, ann_file = ann_file)
        datasets = []
        datasets.append(dataset_temp)
        self.dataset = ConcatDataset(datasets)
                
        #imgs, labels = PACS_SingleDomain.read_txt(self.split_file, self.root_path)
        #self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
        

    
class PACS_FedDG():
    def __init__(self, args, test_domain='C_1', batch_size=4):
        self.batch_size = batch_size
        self.domain_list = list(pacs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = PACS_FedDG.SingleSite(args, domain_name, self.batch_size)
            
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(args, domain_name, batch_size=16):
        dataset_dict = {
            'train': PACS_SingleDomain(args, domain_name=domain_name, split='train_polypGen').dataset,
            'val': PACS_SingleDomain(args, domain_name=domain_name, split='val_polypGen').dataset,
            'test': PACS_SingleDomain(args, domain_name=domain_name, split='val_polypGen').dataset,
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict
    