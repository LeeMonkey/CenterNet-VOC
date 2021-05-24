#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
import cv2
import random
import os.path as osp
import numpy as np
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

def get_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.findtext('xmin'))
        ymin = float(bndbox.findtext('ymin'))
        xmax = float(bndbox.findtext('xmax'))
        ymax = float(bndbox.findtext('ymax'))
        boxes.append([xmin, ymin, xmax, ymax])
    if boxes:
        return np.array(boxes)
    return None 

def main():
    crop_dir = sys.argv[1]
    crop_list = os.listdir(crop_dir)

    image_dir = sys.argv[2]
    image_list = os.listdir(image_dir)

    anno_dir = sys.argv[3]
    iou_thresh = 0.2

    save_dir = 'Temp'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    ratio = 0.5

    for name in image_list:
        if not name.endswith('jpg'):
            continue
        prefix, ext = osp.splitext(name)
        xml_path = osp.join(anno_dir, '{}.xml'.format(prefix))
        if not osp.exists(xml_path):
            continue
        boxes = get_boxes(xml_path)
        if boxes is None:
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
                    crop = cv2.resize(crop, None, fx=0.5, fy=0.5)
                    h, w, c = crop.shape
                    for _ in range(50):
                        top = random.randint(0, height - h)
                        left = random.randint(0, width - w)

                        crop_box = np.array([left, top, left + w, top + h], dtype=boxes.dtype)
                        expand_crop_box = np.tile(crop_box, (boxes.shape[0], 1))
                        max_bbox = np.maximum(expand_crop_box, boxes)
                        min_bbox = np.minimum(expand_crop_box, boxes)
                        area_bbox2 = np.prod(boxes[..., 2:] - boxes[..., :2], axis=-1)
                        iou_bbox = np.concatenate((max_bbox[..., :2], min_bbox[..., 2:]), axis=-1)
                        area_iou_bbox = np.prod(np.maximum(iou_bbox[..., 2:] - iou_bbox[..., :2], 0), axis=-1)
                        iou = area_iou_bbox / area_bbox2
                        if np.max(iou) <= iou_thresh:
                            print('#' * 50)
                            print(iou)
                            image[top:top+h, left:left + w] = cv2.addWeighted(image[top:top+h, left:left + w], 
                                                                              0.1,
                                                                              crop,
                                                                              0.9,
                                                                              0)
                            break
        save_path = osp.join(save_dir, name)
        cv2.imwrite(save_path, image)

if __name__ == '__main__':
    main()
