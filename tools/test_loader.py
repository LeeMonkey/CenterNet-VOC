# -*- coding: utf-8 -*-
'''
Created on Sun Jan  5 13:57:15 2020

@author: Lee
'''
import os
import sys
import os.path as osp

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import numpy as np
import time

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../')))
import src.config as config
from src.modeling.loss import CtdetLoss
from src.dataset import CenterNetDataset
from src.utils import CenterNetTransform, VOCAnnotationTransform, GradualWarmupScheduler
from src.backbone.mobilenetv2 import MobileNet
from src.dataset import DataLoaderX

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

    use_gpu = torch.cuda.is_available()
    index =  torch.cuda.current_device()
    
    device = torch.device('cuda', index) if use_gpu else torch.device('cpu')
    
    # dataset
    train_transform = CenterNetTransform(size=config.INPUT_SIZE, 
                                         mean=config.MEAN, 
                                         std=config.STD, 
                                         is_training=True)
    target_transform = VOCAnnotationTransform() 

    train_dataset = CenterNetDataset('data', 
                               image_sets = [('Armour', 'train'),
                                             ('Car', 'car_train'),
                                             ('Inflatable', 'inflatable_train')], 
                               image_no_boxes=True, 
                               transform=train_transform, 
                               target_transform=target_transform)

    train_loader = DataLoaderX(device, dataset=train_dataset, batch_size=32, shuffle=True, num_workers=32) 
    #train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=32) 
    torch.cuda.synchronize()
    start = time.time()
    for sample in train_loader:
        pass
    torch.cuda.synchronize()
    t = time.time() - start
    print('load time per epoch is {:.3f}s'.format(t))

if __name__ == '__main__':
    main()
