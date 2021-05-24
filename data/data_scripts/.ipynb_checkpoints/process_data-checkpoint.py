#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
from os import path as osp
import shutil
import cv2
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

def merge_xml(xml1, xml2):
    tree1 = ET.parse(xml1)
    tree2 = ET.parse(xml2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    if root1.find('object') is None \
        and root2.find('object') is None:
        return None

    num_boxes = len(root1.findall('object')) + len(root2.findall('object'))
    for obj in root1.iter('object'):
        root2.append(obj)

#     for obj in root2.iter('object'):
#         name = obj.find('name')
#         if name.text == 'armor_cover':
#             name.text = '63'
#         elif name.text == 'tank_cover':
#             name.text = '59'
#         elif name.text == '86_cover':
#             name.text = '86'

    assert num_boxes == len(root2.findall('object'))
    return tree2

def edit_class(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    if root.find('object') is None:
        return None

#     for obj in root.iter('object'):
#         name = obj.find('name')
#         if name.text == 'armor_cover':
#             name.text = '63'
#         elif name.text == 'tank_cover':
#             name.text = '59'
#         elif name.text == '86_cover':
#             name.text = '86'

    return tree

def statistic(save_dir):
    assert osp.isdir(save_dir)
    xml_list = os.listdir(save_dir)
    count = {}
    num = 0
    for xml in xml_list:
        if not xml.endswith('xml'):
            continue
        xml_path = osp.join(save_dir, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            clz = obj.findtext('name')
            if clz not in count:
                count[clz] = 1
            else:
                count[clz] += 1
        num += 1

    print('{split}Statistic{split}'.format(split='-'*60))
    print('num: {:d}'.format(num))
    for k in count:
        print("{}: {:d}".format(k, count[k]))

def edit_xml(tree, size, xml_save_path):
    assert isinstance(size, (tuple, list)) and len(size) == 2
    height, width = size
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
    xml_dir1 = sys.argv[2]
    xml_dir2 = sys.argv[3]
    
#     image_save_dir = 'JPEGImages'
    anno_save_dir = 'labels'
    resize_ratio = 3
    
#     if not osp.isdir(image_save_dir):
#         os.makedirs(image_save_dir)
    
    if not osp.isdir(anno_save_dir):
        os.makedirs(anno_save_dir)
        
    assert osp.isdir(xml_dir1) \
        and osp.isdir(xml_dir2)

    xml_list1 = os.listdir(xml_dir1)
    all_xml = xml_list1[:]
    xml_list2 = os.listdir(xml_dir2)
    all_xml.extend(xml_list2)
    all_xml = set(all_xml)

    for xml in all_xml:
        if not xml.endswith('xml'):
            continue
        prefix, ext = osp.splitext(xml)
        image_path = osp.join(image_dir, '{}.jpg'.format(prefix))

        if not osp.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        if (xml in xml_list1) and (xml in xml_list2):
            xml_path1 = osp.join(xml_dir1, xml)
            xml_path2 = osp.join(xml_dir2, xml)
            tree = merge_xml(xml_path1, xml_path2)
        else:
            xml_path = osp.join(xml_dir1 if xml \
                                in xml_list1 else xml_dir2,
                                xml)
            tree = edit_class(xml_path)
        if tree is not None:
            xml_save_path = osp.join(anno_save_dir, xml)
            tree.write(xml_save_path)
            #image = cv2.resize(image, None, 
            #                fx=1/resize_ratio, fy=1/resize_ratio)
            #height, width, channel = image.shape
            #edit_xml(tree, (height, width), xml_save_path)
            #image_save_path = osp.join(image_save_dir, '{}.jpg'.format(prefix))
            #cv2.imwrite(image_save_path, image)
    #statistic(anno_save_dir)

if __name__ == '__main__':
    main()
