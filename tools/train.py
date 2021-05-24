# -*- coding: utf-8 -*-
'''
Created on Sun Jan  5 13:57:15 2020

@author: Lee
'''
import os
import sys
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../')))
import src.config as config
from src.modeling.loss import CtdetLoss
from src.dataset import CenterNetDataset
from src.utils import CenterNetTransform, VOCAnnotationTransform, GradualWarmupScheduler
from src.backbone.mobilenetv2 import MobileNet

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    use_gpu = torch.cuda.is_available()
    
    model = MobileNet(num_classes=config.NUM_CLASSES)
    print('cuda devices size: {} | current cuda device index: {}'\
        .format(torch.cuda.device_count(), torch.cuda.current_device()))
    
    # model dir
    model_dir = config.MODEL_DIR
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print('=> Save trained model to {}'.format(model_dir))
    
    print('Classes Weight: {}'.format(config.LOSS_WEIGHTS['classes_weight']))
    criterion = CtdetLoss(config.LOSS_WEIGHTS)
    
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    if config.LOAD_PRETRAINED_MODEL:
        weights_path = config.PRETRAINED_MODEL or os.path.join(config.MODEL_DIR, 'last.pth')
        model.load_state_dict(torch.load(weights_path))
        print('=> Load pretrained models: `{}`'.format(weights_path))
    
    model.to(device)

    # dataset
    train_transform = CenterNetTransform(is_training=True)
    test_transform = CenterNetTransform(is_training=False)
    target_transform = VOCAnnotationTransform() 

    train_dataset = CenterNetDataset('data', 
                               image_sets = [('Armour', 'train'),
                                             ('Car', 'car_train'),
                                             ('Inflatable', 'inflatable_train')], 
                               image_no_boxes=True, 
                               transform=train_transform, 
                               target_transform=target_transform)

    test_dataset = CenterNetDataset('data', 
                               image_sets = [('Armour', 'test')], 
                               image_no_boxes=True, 
                               transform=test_transform, 
                               target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32) 
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print('the train dataset has {:d} images,\nthe test dataset has {:d} images'\
        .format(len(train_dataset), len(test_dataset)))
    print('{}\n'.format('#'*60))
    
    # different learning rate
    conv2d_keys = []
    for key, param in model.named_modules():
        if isinstance(param, torch.nn.Conv2d):
            conv2d_keys.append(key + '.weight')
    
    weights_params = []
    others_params = []
    for key, value in model.named_parameters():
        if key in conv2d_keys:
            weights_params.append(value)
        else:
            others_params.append(value)
    
    params = [
        {'params': weights_params, 'weight_decay':1e-4},
        {'params': others_params}
    ]
    
    # optimizer
    optimizer = torch.optim.Adam(params, lr=config.BASE_LR)
    
    # lr_scheduler
    scheduler_multisteplr = MultiStepLR(optimizer, 
                                        milestones=config.ADJUST_LR_EPOCHS, 
                                        gamma=config.GAMMA)

    scheduler = GradualWarmupScheduler(optimizer, 
                                       multiplier=1, 
                                       total_epoch=config.WARMUP_EPOCHS, 
                                       after_scheduler=scheduler_multisteplr)
    
    best_test_loss = np.inf 
    num_iters_per_epoch = len(train_loader)
    num_epochs = config.NUM_EPOCHS
    start_epoch = 0
    
    for epoch in range(start_epoch, num_epochs):
    
        total_loss = 0.
        
        model.train()
        for i, sample in enumerate(train_loader):
            # input to device 
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
            pred = model(sample['input'])
            hm_loss, wh_loss, off_loss, loss = criterion(pred, sample)    
            total_loss += loss.item()

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0 or (i+1) == num_iters_per_epoch:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f %s' 
                %(epoch+1, num_epochs, i+1, len(train_loader), 
                    loss.data, total_loss / (i+1), ' '.join(['pg{:d}_lr: {:.6f}'.format(idx, pg['lr']) \
                            for idx, pg in enumerate(optimizer.param_groups)])))
                print('\__hm_loss:{:.4f} wh_loss: {:.4f} off_loss:{:.4f}'.format(hm_loss.item(), wh_loss.item(), off_loss.item()))
        scheduler.step(epoch + 1)
    
        #validation
        hm_vloss = wh_vloss = off_vloss = validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                for k in sample:
                    sample[k] = sample[k].to(device=device, non_blocking=True)
                pred = model(sample['input'])
                hm_loss, wh_loss, off_loss, loss = criterion(pred, sample)
                validation_loss += loss.item()
                hm_vloss += hm_loss.item()
                wh_vloss += wh_loss.item()
                off_vloss += off_loss.item()

            validation_loss /= len(test_loader)
            hm_vloss /= len(test_loader)
            wh_vloss /= len(test_loader)
            off_vloss /= len(test_loader)
        
            print('Epoch [{:d}/{:d}], validation loss: {:.4f}'.format(epoch + 1, num_epochs, validation_loss))
            print('\__hm_loss: {:.4f} wh_loss: {:.4f} off_loss:{:.4f}'.format(
                hm_vloss, wh_vloss, off_vloss))

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, 'last.pth'))

if __name__ == '__main__':
    main()
