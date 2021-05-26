# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lee
"""
import os
import sys

import torch
import torch.nn as nn

import os.path as osp
import cv2
import numpy as np
import time

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../')))
import src.config as config
from src.backbone.mobilenetv2 import MobileNet
from src.dataset import CenterNetDataset
from src.utils import CenterNetTransform, VOCAnnotationTransform
from src.modeling.loss import _gather_feat, _transpose_and_gather_feat

def draw(img, res, is_gt=False):
    draw_image = img.copy()
    if is_gt:
        if res is None:
            return

    for box in res:
        xmin, ymin, xmax, ymax = box[:4]
        label_str = config.CENTERNET_CLASSES[int(box[-1])]

        if 'colors' not in globals():
            globals()['colors'] = {}
        if label_str not in globals()['colors']:
            r = np.random.randint(50, 255)
            g = np.random.randint(150, 255)
            b = np.random.randint(100, 255)
            globals()['colors'][label_str] = (b, g, r)

        color = globals()['colors'][label_str]
        if is_gt:
            color = tuple([255 - v for v in color])
        draw_image = cv2.rectangle(draw_image,
                                   (int(xmin), int(ymin)),
                                   (int(xmax), int(ymax)),
                                   color,
                                   2)
        if not is_gt:
            score = box[4]
            draw_image = cv2.putText(draw_image,
                        "{:s}: {:.2f}".format(label_str, score),
                        (int(xmin), int(ymin) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)
    return draw_image


def _nms(heat, kernel=3):
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


def decode(heat, wh, reg=None, K=40):
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
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
      output = model(images)
      hm = output['heatmap'].sigmoid_()
      wh = output['wh']
      reg = output['offset'] 
      dets = decode(hm, wh, reg=reg, K=100) # K 是最多保留几个目标
      torch.cuda.synchronize()
      forward_time = time.time() - start_time
    if return_time:
      return dets, forward_time
    else:
      return dets


def get_rescale_ratio(src_size, dst_size):
    assert len(src_size) == len(dst_size) \
        and len(src_size) == 2
    sw, sh = src_size
    dw, dh = dst_size
    return max(dh / sh, dw / sw)


def postprocess(dets, src_size, dst_size):
    ratio = get_rescale_ratio(src_size, dst_size) * config.DOWNSAMPLE_RATIO
    dets = dets.detach().cpu().numpy().reshape(-1, dets.shape[-1])
    dets[..., :4] *= ratio
    return dets


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


def apply_nms(dets, hm_thresh=0.3, nms_thresh=0.3, global_nms=True):
    dets = dets[dets[:, 4] >= hm_thresh]
    if len(dets):
        xmin = dets[:, 0]
        ymin = dets[:, 1]
        xmax = dets[:, 2]
        ymax = dets[:, 3]
        scores = dets[:, 4]
        areas = (xmax -  xmin + 1) * (ymax - ymin + 1)
        desc = np.argsort(-scores)

        keep = np.array([True] * desc.size)
        for i, idx in enumerate(desc):
            if not keep[idx]:
                continue

            mask = keep.copy()
            mask[:(i+1)] = False
            if not global_nms:
                mask &= (dets[:, -1] == dets[idx][-1])

            if not any(mask):
                break
                
            left = np.maximum(xmin[idx], xmin[desc[mask]])
            top = np.maximum(ymin[idx], ymin[desc[mask]])
            right = np.minimum(xmax[idx], xmax[desc[mask]])
            bottom = np.minimum(ymax[idx], ymax[desc[mask]])
            w = np.maximum(0, right - left + 1)
            h = np.maximum(0, bottom - top + 1)
            intersect_areas = w * h
            iou = intersect_areas / (areas[idx] + areas[desc[mask]] - intersect_areas)
            keep[mask] = (iou <= nms_thresh)
        dets = dets[keep]
    return dets
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    infer_images_dir = 'data/Armour/JPEGImages'
    model_path = osp.join('models/Armour/20210524/640x640', 'last.pth')
    test_sets = [('Armour', 'test')]
    output_dir = "img_ret/Armour"

    hm_thresh = 0.3
    nms_thresh = 0.3

    # init transform
    transform = CenterNetTransform(size=config.INPUT_SIZE,
                                        mean=config.MEAN,
                                        std=config.STD,
                                        is_training=False)

    target_transform = VOCAnnotationTransform(class_to_ind=None,
                                              keep_difficult=False)

    test_dataset = CenterNetDataset('data', 
                               image_sets = test_sets, 
                               image_no_boxes=True, 
                               transform=None, 
                               target_transform=target_transform)


    # create net and load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    model = MobileNet(pretrained=False, num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    for index in range(len(test_dataset)):
        image, target = test_dataset.pull_item(index)

        print('-' * 40)
        # preprocessing
        tran_image, _ = transform(image)
        tran_image = tran_image.transpose(2, 0, 1)
        tran_image = tran_image[np.newaxis, ...]
        tran_image = torch.from_numpy(tran_image)
        tran_image = tran_image.to(device)

        # forward
        dets, forward_time = process(tran_image, return_time=True)
        print('forward time: {:.2f}ms'.format(forward_time * 1000))
        src_height = tran_image.size(2)
        src_width = tran_image.size(3)

        dst_height, dst_width = image.shape[:2]
        # post process
        start = time.time()
        dets = postprocess(dets, (src_width, src_height), (dst_width, dst_height))
        dets = apply_nms(dets, hm_thresh, nms_thresh)
        post_time = time.time() - start
        print('post processing time: {:.2f}ms'.format(post_time * 1000))

        draw_image = draw(image, dets, is_gt=False)
        #draw_image = draw(image, target, is_gt=True)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(index)), draw_image)
