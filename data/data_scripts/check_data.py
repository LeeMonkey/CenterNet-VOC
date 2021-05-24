#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
import cv2
import os.path as osp
import argparse
import json
import random
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check the legitimacy of the data')
    parser.add_argument('--image_dir', dest='image_dir', type=str, default='JPEGImages', 
                        help='the dir of input image')
    parser.add_argument('--anno_dir', dest='anno_dir', type=str, default='Annotations', 
                        help='the dir of input anno')

    if len(sys.argv) <= 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def parse_xml(xml_path, pre_id, pre_image_id, categories, is_coco=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annos = []
    for obj in root.iter('object'):
        clz = obj.findtext('name')
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            anno = {}
            xmin = int(bndbox.findtext('xmin'))
            ymin = int(bndbox.findtext('ymin'))
            xmax = int(bndbox.findtext('xmax'))
            ymax = int(bndbox.findtext('ymax'))
            anno['bbox'] = [(xmin + xmax) / 2 if not is_coco else xmin, 
                            (ymin + ymax) / 2 if not is_coco else ymin,
                            xmax - xmin,
                            ymax - ymin]
            anno['area'] = (xmax - xmin) * (ymax - ymin)
            if clz not in categories:
                categories.append(clz)
            anno['category_id'] = categories.index(clz) + 1
            if annos:
                anno['id'] = annos[-1]['id'] + 1
            else:
                anno['id'] = pre_id + 1
            anno['image_id'] = pre_image_id + 1
            anno['iscrowd'] = 0
            annos.append(anno)
    return annos

def main():
    args = parse_arguments()
    image_dir = args.image_dir
    anno_dir = args.anno_dir
    assert osp.isdir(image_dir), '`{}` not exists'.format(image_dir)
    assert osp.isdir(anno_dir), '`{}` not exists'.format(anno_dir)

    ratio_train = 0.9
    image_id = 0
    anno_id = 0
    categories = []

    all_annos = {}
    for name in os.listdir(image_dir):
        if random.random() <= ratio_train:
            phase = ['train']
        else:
            phase = ['test', 'coco_test']
        for k in phase:
            if k not in all_annos:
                all_annos[k] = {}
            coco_json = all_annos[k]
            prefix, suffix = osp.splitext(name)
            anno_path = osp.join(anno_dir, '{}.xml'.format(prefix))
            if not name.endswith('.jpg') or (not osp.exists(anno_path)):
                break
            image_path = osp.join(image_dir, name)
            annos = parse_xml(anno_path, anno_id, image_id, categories, is_coco=True if k == 'coco_test' else False)
            if annos:
                if 'images' not in coco_json:
                    coco_json['images'] = []
                if 'annotations' not in coco_json:
                    coco_json['annotations'] = []
                image = cv2.imread(image_path)
                if image is None:
                    continue
                h, w, c = image.shape
                coco_json['images'].append({
                    'id':annos[-1]['image_id'],
                    'width':w,
                    'height':h,
                    'file_name':name
                })
                coco_json['annotations'].extend(annos)
                image_id = annos[-1]['image_id']
                anno_id = annos[-1]['id']

    for k in all_annos:
        coco_json = all_annos[k]
        for i, cate in enumerate(categories, 1):
            if 'categories' not in coco_json:
                coco_json['categories'] = []
            coco_json['categories'].append({
                'id':i,
                'name': cate,
                'supercategory':'armoured_equipment'
            })
        with open("{}.json".format(k), 'w') as f:
            json.dump(coco_json, f, indent=2)
    
        
if __name__ == '__main__':
    main()
