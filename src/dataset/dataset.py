#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
import os.path as osp
import numpy as np
import cv2
import torch.utils.data as data
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../')))
from src.config import *
from src.utils import draw_umich_gaussian, gaussian_radius 

class ConvertHeatmap:
    def __init__(self, num_classes, max_objs=128, downsampling_rate=4):
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.downsampling_rate = downsampling_rate

    def __call__(self, image, target=None):
        _, height, width = image.shape
        output_w = width // self.downsampling_rate
        output_h = height // self.downsampling_rate

        heatmap = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        indexes = np.zeros((self.max_objs), dtype=np.int64)
        offset_mask = np.zeros((self.max_objs), dtype=np.int8)
        if target is not None:
            for i, bbox in enumerate(target.copy()):
                if i >= self.max_objs:
                    break
                bbox[:4] /= self.downsampling_rate
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1] 
                if w > 0 and h > 0:
                    ceil_w = int(w) + 1 if w % 1 else int(w)
                    ceil_h = int(h) + 1 if h % 1 else int(h)
                    radius = gaussian_radius((ceil_h, ceil_w))
                    radius = max(0, int(radius))
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    center_int = center.astype(np.int32)
                    draw_umich_gaussian(heatmap[int(bbox[-1])], center_int, radius)
                    wh[i] = w, h 
                    indexes[i] = center_int[1] * output_w + center_int[0]
                    offset[i] = center - center_int
                    offset_mask[i] = 1

        return {'input': image,
                'heatmap': heatmap,
                'wh': wh,
                'offset': offset,
                'indexes': indexes,
                'offset_mask': offset_mask}

    def print_result(self, image, save_path=None, target=None, heatmap=None):
        if save_path is None:
            return
        draw_image = image.transpose(1, 2, 0).copy().astype(np.uint8)
        if target is not None:
            for bbox in target:
                cv2.rectangle(draw_image,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 0, 255),
                              2)
                cv2.putText(draw_image,
                            '{:d}'.format(int(bbox[-1])),
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)
        if heatmap is not None:
            ih, iw, _ = draw_image.shape
            draw_heatmap = heatmap.copy()
            draw_heatmap = draw_heatmap.transpose(1, 2, 0)[..., np.newaxis].astype(np.float32)
            draw_heatmap = (draw_heatmap * np.array([[0, 255, 255]], dtype=np.float32)).max(axis=2)
            draw_heatmap = draw_heatmap.astype(np.uint8)
            draw_heatmap = cv2.resize(draw_heatmap, (iw, ih))
            fore = 255 - draw_heatmap
            if len(fore.shape) == 2:
                fore = fore.reshape(*fore.shape, 1)
            draw_image = cv2.addWeighted(draw_image, 0.3, fore, 0.7, 0)
        cv2.imwrite(save_path, draw_image)
                

class CenterNetDataset(data.Dataset):
    def __init__(self, root, 
                 image_sets = [('GTZ', 'train')],
                 image_no_boxes=True,
                 transform=None, 
                 target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform =  target_transform
        self.image_path_fmt = osp.join('{}', 'JPEGImages', '{}.jpg') 
        self.xml_path_fmt = osp.join('{}', 'Annotations', '{}.xml')
        self.heatmap_transform = ConvertHeatmap(num_classes=NUM_CLASSES)

        assert len(image_sets), 'image_sets is empty!'

        self.ids = list()
        for folder, name in image_sets:
            data_dir = osp.join(root, folder)
            assert osp.isdir(data_dir)
            list_path = osp.join(data_dir, 
                                 'ImageSets', 
                                 'Main', 
                                 '{}.txt'.format(name))
            assert osp.exists(list_path)
            with open(list_path) as f:
                for line in f:
                    prefix = line.strip()
                    image_path = self.image_path_fmt.format(data_dir, prefix)
                    xml_path = self.xml_path_fmt.format(data_dir, prefix)
                    if not osp.exists(image_path):
                        raise FileNotFoundError('No such image file: \'{}\''.format(image_path))
                    if osp.exists(xml_path):
                        self.ids.append((image_path, xml_path))
                    elif image_no_boxes:
                            self.ids.append((image_path, None))
                    else:
                        raise FileNotFoundError('No such xml file: \'{}\''.format(xml_path))

    def __getitem__(self, index):
        image, gt = self.pull_item(index)
        data = self.heatmap_transform(image, gt)
        # print result
        #self.heatmap_transform.print_result(data['input'],
        #                                    save_path='1.jpg',
        #                                    target=gt,
        #                                    heatmap=data['heatmap'])

        return data

    def pull_item(self, index):
        image_path, xml_path = self.ids[index]
        image = cv2.imread(image_path)
        if self.target_transform is not None:
            if xml_path is not None:
                target = self.target_transform(xml_path)
            else:
                target = None

        if self.transform is not None:
            target = np.array(target) if target else target
            image, target = self.transform(image, target)
            image = image.transpose(2, 0, 1)
        return image, target 

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    from src.utils import CenterNetTransform, VOCAnnotationTransform  
    transform = CenterNetTransform(is_training=False)
    target_transform = VOCAnnotationTransform()
    dataset = CenterNetDataset('data', 
                               image_sets = [('Armour', 'train')], 
                               image_no_boxes=True, 
                               transform=transform, 
                               target_transform=target_transform)
    for _ in dataset:
        pass
