# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lee
"""
import os
import sys
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from pycocotools.coco import COCO
from dataset import ctDataset
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET


sys.path.append("backbone")
#from resnet_dcn import ResNet
#from dlanet_dcn import DlaNet
from mobilenetv2 import MobileNet

from Loss import _gather_feat
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat

def draw(img, res, is_coco=False):
    #labels = {1:'Type_63_Tracked_Armoured_Vehicle', 
    #        2:'Type_59_Medium_Tank',
    #        3:'ZBD-86_Infantry_Fighting_Vehicle'}
    labels = {1:'59', 
            2:'63',
            3:'86',
            4:'08'}
    draw_image = img.copy()
    for ann in res:
        if is_coco:
            lx = int(ann["bbox"][0] - ann["bbox"][2] / 2)
            ly = int(ann["bbox"][1] - ann["bbox"][3] / 2)
            rx = int(ann["bbox"][0] + ann["bbox"][2] / 2)
            ry = int(ann["bbox"][1] + ann["bbox"][3] / 2)
            label = ann['category_id']
        else:
            lx, ly, rx, ry = [int(x) for x in ann[:-2]]
            score = ann[-2]
            label = int(ann[-1])

        if 'colors' not in globals():
            globals()['colors'] = {}
        if label not in globals()['colors']:
            r = np.random.randint(50, 255)
            g = np.random.randint(150, 255)
            b = np.random.randint(100, 255)
            globals()['colors'][label] = (b, g, r)
        draw_image = cv2.rectangle(draw_image,
                      (lx, ly),
                      (rx, ry),
                      globals()['colors'][label],
                      2)

        if not is_coco:
            draw_image = cv2.putText(draw_image,
                        "{:s}: {:.2f}".format(labels[label], score),
                        (lx, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        globals()['colors'][label],
                        2)
    return draw_image

def pre_process(image):
    height, width = image.shape[0:2]
    #inp_height, inp_width = 512, 512
    #inp_height = inp_width = 608
    inp_height, inp_width = ctDataset.default_resolution
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)

    """
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
    
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)
    """
    mean = ctDataset.mean
    std = ctDataset.std
    
    inp_image = (inp_image.astype(np.float32) / 255. - mean) / std

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width) # 三维reshape到4维，（1，3，512，512） 
    
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // 4, 
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=5):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float() 
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
   
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    
    """
    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, ang.shape[-1])
    """

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def process(images, return_time=False):
    start_time = time.time()
    with torch.no_grad():
      output = model(images)
      hm = output['hm'].sigmoid_()
      #ang = output['ang'].relu_()
      wh = output['wh']
      reg = output['reg'] 
      torch.cuda.synchronize()
      forward_time = time.time() - start_time
      dets = ctdet_decode(hm, wh, reg=reg, K=100) # K 是最多保留几个目标
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  
    num_classes = ctDataset.num_classes
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = ctDataset.num_classes
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
      kth = len(scores) - max_obj_per_img
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, num_classes + 1):
        keep_inds = (detections[j][:, 4] >= thresh)
        detections[j] = detections[j][keep_inds]
    return detections


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "

        if not elem.tail or not elem.tail.strip():
            elem.tail = i

        for elem in elem:
            indent(elem, level+1)

        if not elem.tail or not elem.tail.strip():
            elem.tail = i 
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i 

def process_frame(file_name, frame, labels, save_dir="JPEGImages"):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = save_dir
    filename = ET.SubElement(root, "filename")
    filename.text = file_name

    #size
    size = ET.SubElement(root, "size")
    depth = ET.SubElement(size, "depth")
    depth.text = "{}".format(frame.shape[-1])
    width = ET.SubElement(size, "width")
    width.text = "{}".format(frame.shape[1])
    height = ET.SubElement(size, "height")
    height.text = "{}".format(frame.shape[0])

    for cls_id, x1, y1, x2, y2, score in labels:
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        name.text = "car"
        bndbox = ET.SubElement(object, "bndbox")

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(x1)
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(y1)
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(x2)
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(y2)

        difficult = ET.SubElement(bndbox, "difficult")
        difficult.text = "0" 
    
    indent(root)
    tree = ET.ElementTree(root)
    xml_path = os.path.join("labels", os.path.splitext(file_name)[0] + ".xml")
    tree.write(xml_path, encoding="utf-8")

def py_cpu_nms(all_dets, thresh=0.45, global_nms=True):
    """Pure Python NMS baseline."""
    nms_dets = []
    if len(all_dets) == 0:
        return nms_dets
    if global_nms:
        dets = all_dets
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h 
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        if keep:
            nms_dets.append(dets[keep])
    else:
        for cls in range(1, ctDataset.num_classes+1):
            dets = all_dets[all_dets[:, -1] == cls]
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h 
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]
            if keep:
                nms_dets.append(dets[keep])
    if nms_dets:
        nms_dets = np.concatenate(nms_dets, axis=0)
    return nms_dets
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    infer_images_dir = 'data/Armour/JPEGImages'
    model_path = "models/Armour/20210519/640x640/last.pth"
    anno_file = "data/Armour/test.json"
    output_dir = "img_ret/Armour"

    hm_thresh = 0.3
    nms_thresh = 0.3

    #model = ResNet(18)
    model = MobileNet(pretrained=False, num_classes=ctDataset.num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    import random
    random.shuffle(img_ids)

    for iid in img_ids:
        img_anno = coco.loadImgs(iid)[0]
        file_name = img_anno["file_name"] 

        ann_ids = coco.getAnnIds(imgIds=img_anno['id'], iscrowd=None)
        coco_anns = coco.loadAnns(ann_ids)

        if file_name.split('.')[-1] == 'jpg':
            image_path = os.path.join(infer_images_dir, file_name)
            image = cv2.imread(image_path)
            images, meta = pre_process(image)
            images = images.to(device)
            output, dets, forward_time = process(images, return_time=True)
            print("forward time: {:.2f}ms\n{}".format(forward_time * 1000, "-" * 40))
            
            dets = post_process(dets, meta)
            ret = merge_outputs(dets)

            dets = []
            for i, c in ret.items():
                mask = c[:, 4] > hm_thresh
                if mask.any():
                    c = c[mask]
                    cls = np.full((c.shape[0], 1), i, dtype=c.dtype)
                    c = np.concatenate((c, cls), axis=-1)
                    dets.append(c)

            if len(dets) == 1:
                dets = dets[0]
            elif len(dets) > 1:
                dets = np.concatenate(dets)
                
            dets = py_cpu_nms(dets, nms_thresh)
            
            draw_image = draw(image, dets, is_coco=False)  # 画旋转矩形
            #draw_image = draw(draw_image, coco_anns, is_coco=True)  # 画旋转矩形
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(os.path.join(output_dir, file_name), draw_image)
