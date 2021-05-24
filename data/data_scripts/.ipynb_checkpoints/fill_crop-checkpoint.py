#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
import cv2
import random
import os.path as osp

def main():
    crop_dir = sys.argv[1]
    crop_list = os.listdir(crop_dir)

    image_dir = sys.argv[2]
    image_list = os.listdir(image_dir)

    save_dir = 'temp'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    ratio = 0.3

    for name in image_list:
        if not name.endswith('jpg'):
            continue
        image_path = osp.join(image_dir, name)
        image = cv2.imread(image_path)
        if random.random() < ratio:
            height, width, channel = image.shape
            if width > 800:
                num = random.randint(1, len(crop_list) // 2 + 1)
                pick_crop_list = random.sample(crop_list, num)
                for f in pick_crop_list:
                    if not f.endswith('jpg'):
                        continue
                    crop_path = osp.join(crop_dir, f)
                    crop = cv2.imread(crop_path)
                    h, w, c = crop.shape
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)

                    image[top:top+h, left:left + w] = cv2.addWeighted(image[top:top+h, left:left + w], 
                                                                      0.4,
                                                                      crop,
                                                                      0.7,
                                                                      0)
        save_path = osp.join(save_dir, name)
        cv2.imwrite(save_path, image)

if __name__ == '__main__':
    main()
