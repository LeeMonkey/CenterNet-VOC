#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
from os import path as osp
import shutil
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import cv2

def edit_xml(xml_path, size, xml_save_path):
    assert isinstance(size, (tuple, list)) and len(size) == 2
    height, width = size
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return
    owidth = size.findtext('width')
    oheight = size.findtext('height')
    if owidth is None or oheight is None:
        return
    size.find('width').text = str(width)
    size.find('height').text = str(height)
    for obj in root.iter('object'):
        name = obj.find('name')
        if name.text == 'armor_cover':
            name.text = '63'
        elif name.text == 'tank_cover':
            name.text = '59'
        elif name.text == '86_cover':
            name.text = '86'
        
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            for key in ['xmin',  
                        'xmax']:
                item = bndbox.find(key)
                item.text = '{:.2f}'.format(float(item.text) \
                / float(owidth) * width)

            for key in ['ymin',
                        'ymax']:
                item = bndbox.find(key)
                item.text = '{:.2f}'.format(float(item.text) \
                / float(oheight) * height)
    tree.write(xml_save_path)

def main():
    image_dir = sys.argv[1]
    anno_dir = sys.argv[2]
    image_save_dir = 'JPEGImages'
    anno_save_dir = 'Annotations'
    if not osp.isdir(image_save_dir):
        os.makedirs(image_save_dir)

    if not osp.isdir(anno_save_dir):
        os.makedirs(anno_save_dir)

    resize_ratio = 1
    xml_list = os.listdir(anno_dir)
    for xml in xml_list:
        prefix, ext = osp.splitext(xml)
        xml_path = osp.join(anno_dir, xml)
        image_path = osp.join(image_dir, '{}.jpg'.format(prefix))
        if not osp.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, None, 
                            fx=1/resize_ratio, fy=1/resize_ratio)
            height, width, channel = image.shape
            image_save_path = osp.join(image_save_dir, '{}.jpg'.format(prefix))
            xml_save_path = osp.join(anno_save_dir, xml)
            edit_xml(xml_path, (height, width), xml_save_path)
            cv2.imwrite(image_save_path, image)

if __name__ == '__main__':
    main()
