#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
from os import path as osp
import cv2
import numpy as np
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import shutil

def merge_xml(light_xml, infrared_xml, rescale_size, top_left):
    tree1 = ET.parse(light_xml)
    tree2 = ET.parse(infrared_xml)
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    anno1 = []
    for obj in root1.iter('object'):
        name = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        anno1.append({'name':name,
                       'box':[xmin, ymin, xmax, ymax]})
    if len(anno1) == 0:
        print(light_xml)

    anno2 = []
    size = root2.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    wratio = rescale_size[0] / width
    hratio = rescale_size[1] / height
    for obj in root2.iter('object'):
        name = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax')) 
        anno2.append({'name':name,
                       'box':[xmin, ymin, xmax, ymax]})
    if len(anno2) == 0:
        print(infrared_xml)
    
    bbox1 = np.array([l['box'] for l in anno1], dtype=np.float64)
    bbox2 = np.array([l['box'] for l in anno2], dtype=np.float64)
    bbox2[:, ::2] *= wratio
    bbox2[:, ::2] += top_left[0]
    bbox2[:, 1::2] *= hratio
    bbox2[:, 1::2] += top_left[1]

    expand_bbox1 = np.tile(bbox1, [bbox2.shape[0], 1, 1])
    expand_bbox2 = np.repeat(bbox2[:, np.newaxis], bbox1.shape[0], axis=1)

    max_bbox = np.maximum(expand_bbox1, expand_bbox2)
    min_bbox = np.minimum(expand_bbox1, expand_bbox2)
    # w, h = (xmax, ymax) - (xmin, ymin) && w * h
    area_bbox1 = np.prod(expand_bbox1[..., 2:] - expand_bbox1[..., :2], axis=-1)
    area_bbox2 = np.prod(expand_bbox2[..., 2:] - expand_bbox2[..., :2], axis=-1)
    iou_bbox = np.concatenate((max_bbox[..., :2], min_bbox[..., 2:]), axis=-1)
    area_iou_bbox = np.prod(np.maximum(iou_bbox[..., 2:] - iou_bbox[..., :2], 0), axis=-1)
    iou = area_iou_bbox / ((area_bbox1 + area_bbox2) - area_iou_bbox)
    max_iou_idx = np.argmax(iou, axis=1)
    max_iou_sort = np.argsort(-np.max(iou, axis=1))
    bbox1_selected = [False] * bbox1.shape[0]

    del_list = []
    for i in max_iou_sort:
        if iou[i, max_iou_idx[i]] > 0:
            if not bbox1_selected[max_iou_idx[i]]:
                anno2[i]['name'] = anno1[max_iou_idx[i]]['name']
                bbox1_selected[max_iou_idx[i]] = True
            else:
                anno2[i] = None
        else:
            anno2[i] = None
            del_list.append(i)
    assert len(anno2) == len(root2.findall('object'))

    for ann, obj in zip(anno2, root2.findall('object')):
        if ann is None:
            root2.remove(obj)
            print(infrared_xml)
        else:
            obj.find('name').text = ann['name']
            #print(ann['name'], obj.findtext('name'))

    return tree2
            
def main():
    infrared_images_dir = sys.argv[1]
    infrared_dir = sys.argv[2]
    light_dir = sys.argv[3]
    save_dir = 'infrared_annos'

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    sp = (791, 195)
    ep = (3068, 2034)
    infrared_list = os.listdir(infrared_dir)
    for name in infrared_list:
        if not name.endswith('xml'):
            continue

        device, vid, iid = name.split('_')
        size = len(vid)
        format_str = "{{:0>{}d}}".format(size)
        iname = "_".join([device, format_str.format(int(vid) - 1), iid])
        light_path = osp.join(light_dir, iname)
        if not osp.exists(light_path):
            continue
        image_path = osp.join(infrared_images_dir, '{}.jpg'.format(osp.splitext(name)[0]))
        if not osp.exists(image_path):
            continue
        infrared_path = osp.join(infrared_dir, name)
        rescale_size = tuple(e - s for s, e in  zip(sp, ep))

        tree = merge_xml(light_path, infrared_path, rescale_size, sp)
        tree.write(os.path.join(save_dir, name))
        shutil.copy(image_path, save_dir)

def align_image():
    infrared_dir = sys.argv[1]
    light_dir = sys.argv[2]
    save_dir = 'save'

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    sp = (791, 195)
    ep = (3068, 2034)
    infrared_list = os.listdir(infrared_dir)
    #cv2.namedWindow('alignment', cv2.WINDOW_FREERATIO)
    for name in infrared_list:
        if not name.endswith('jpg'):
            continue

        device, vid, iid = name.split('_')
        size = len(vid)
        format_str = "{{:0>{}d}}".format(size)
        iname = "_".join([device, format_str.format(int(vid) - 1), iid])
        light_path = osp.join(light_dir, iname)
        print(light_path)
        if not osp.exists(light_path):
            continue
        infrared_path = osp.join(infrared_dir, name)
        rescale_size = tuple(e - s for s, e in  zip(sp, ep))

        infrared_image = cv2.imread(infrared_path)
        infrared_image = cv2.resize(infrared_image, rescale_size)
        light_image = cv2.imread(light_path)
        weight_image = cv2.addWeighted(light_image[sp[1]:ep[1], sp[0]:ep[0]], 0, infrared_image, 1, 0)
        light_image[sp[1]:ep[1], sp[0]:ep[0]] = weight_image
        save_path = osp.join(save_dir, name)
        cv2.imwrite(save_path, light_image)
        #cv2.imshow('alignment', light_image)
        #while True:
        #    key = cv2.waitKey(0) 
        #    if key & 0xFF == ord('n'):
        #        break
        #    if key & 0xFF == ord('q'):
        #        cv2.destroyAllWindows()
        #        exit(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    #main()
    align_image()
