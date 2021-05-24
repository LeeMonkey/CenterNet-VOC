# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:57:15 2020

@author: Lee
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ctDataset

train_dataset = ctDataset(split='train')
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=False,num_workers=8)  # num_workers是加载数据（batch）的线程数目

test_dataset = ctDataset(split='test')
test_loader = DataLoader(test_dataset,batch_size=4, shuffle=False,num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))

for i, sample in enumerate(train_loader):
    if i % 10 == 0:
        print("Loading No.{} batches train images".format(i + 1))
#validation
for i, sample in enumerate(test_loader):
    if i % 10 == 0:
        print("Loading No.{} batches test images".format(i + 1))
