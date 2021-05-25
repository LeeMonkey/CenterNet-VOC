#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
import os.path as osp
import time

ROOT=osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT)

# load pretrained model
LOAD_PRETRAINED_MODEL=True
PRETRAINED_MODEL=None

#input size
INPUT_SIZE = (640, 640)
MEAN = (107, 114, 123)
STD = (1, 1, 1)

date = time.strftime('%Y%m%d', time.localtime())
# dir of model
MODEL_DIR='models/Armour/{}/{}x{}'.format(date, *INPUT_SIZE)

# learning rate
BASE_LR=1.25e-3
GAMMA=0.1 # power of adjust lr

# epoch
NUM_EPOCHS=800
WARMUP_EPOCHS=50
ADJUST_LR_EPOCHS = (200, 400, 600)

# data
NUM_CLASSES=4
CENTERNET_CLASSES=('59', '63', '86', '08')
assert NUM_CLASSES == len(CENTERNET_CLASSES)

LOSS_WEIGHTS = {
    'heatmap_weight': 3,
    'wh_weight': 0.8,
    'offset_weight': 0.2,
    'classes_weight': [2, 5, 23, 8]
}
