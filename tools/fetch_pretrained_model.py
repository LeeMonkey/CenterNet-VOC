# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lee
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import sys
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

def main():
    coco_weights = torch.load('models/coco/best.pth')
    base_weights = OrderedDict() 
    keys = coco_weights.keys()
    for k,v in coco_weights.items():
        if "base" in k:
            k = k[5:]
            base_weights[k] = v

    torch.save(base_weights, "models/coco/mobilenetv2_backbone.pth")

if __name__ == '__main__':
    main()
            
