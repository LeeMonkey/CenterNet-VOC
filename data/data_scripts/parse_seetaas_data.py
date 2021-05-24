#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import os.path as osp
from kpl_dataset import ReadOnlyDataset, WriteOnlyDataset, BasicType
import kpl_helper as helper
import sys
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
    
def pretty_xml(element, indent, newline, level=0):
    if element:
        if (element.text is None) or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + \
                element.text.strip() + newline + indent * (level + 1)
        
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)

def main():
    data_path = sys.argv[1]
    save_dir = 'labels'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    reader = ReadOnlyDataset(data_path, data_name=None)
    # method 1. iteration
    for record in reader:
        objs = record['object']
        contentType = record['contentType']
        if objs and contentType == 'image/jpeg':
            width = record['width']
            height = record['height']
            depth = record['depth']
            path = record['path']
            content = record['content']
            image_bts = bytes(content)
            if not osp.isdir(osp.dirname(path)):
                os.makedirs(osp.dirname(path))
            if not osp.exists(path):
                with open(path, 'wb') as f:
                    f.write(image_bts)
                    
                   
            root = ET.Element('annotation')
            p = ET.SubElement(root, 'path')
            p.text = path
            size = ET.SubElement(root, 'size')
            w = ET.SubElement(size, 'width')
            w.text = str(width)
            h = ET.SubElement(size, 'height')
            h.text = str(height)
            d = ET.SubElement(size, 'depth')
            d.text = str(depth)
            for obj in objs:
                object = ET.SubElement(root, 'object')
                name = ET.SubElement(object, 'name')
                name.text = obj['name']
                pose = ET.SubElement(object, 'pose')
                pose.text = 'Unspecified'
                difficult = ET.SubElement(object, 'difficult')
                difficult.text = '0'  
                
                bndbox = ET.SubElement(object, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(obj['xmin'])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(obj['ymin'])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(obj['xmax'])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(obj['ymax'])
            pretty_xml(root, '  ', '\n')
            tree = ET.ElementTree(root)
            _, name = osp.split(path)
            prefix, ext = osp.splitext(name)
            save_path = osp.join(save_dir, '{}.xml'.format(prefix))
            tree.write(save_path)

if __name__ == '__main__':
    main()