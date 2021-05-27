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


def do_python_eval(dets, annos, ovthresh=0.5, use_07=True):
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    aps = []
    print('--------------------------------------------------------------')
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for i, cls in enumerate(config.CENTERNET_CLASSES):
        rec, prec, ap = voc_eval(dets, annos, i, ovthresh=ovthresh, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP @[ IoU={:.2f} ] for {} = {:.4f}'.format(ovthresh, cls, ap))
    print('Mean AP @[ IoU={:.2f} ] = {:.4f}'.format(ovthresh, np.mean(aps)))
    print('~~~~~~~~')
    print('Results @[ IoU={:.2f} ]:'.format(ovthresh))
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_eval(dets,
             annos,
             cls_id,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
dets: detections
annopath: annotations
cls_id: Category index (duh)
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# first load gt
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for iid in annos:
        if annos[iid] is not None:
            mask_anno = annos[iid][annos[iid][:, -1] == cls_id]
            boxes = np.array([anno[:4] for anno in mask_anno])
            class_recs[iid] = {'bbox': boxes,
                               'det': [False] * len(boxes)}
            npos += len(boxes)
        else:
            class_recs[iid] = None
            npos += 1

    # read dets
    if len(dets):
        tp = list()
        fp = list()
        BB = list()
        image_ids = list()
        count = 0

        for iid in dets:
            if len(dets[iid]) == 0:
                if class_recs[iid] is None:
                    tp.append(1.)
                    fp.append(0.)
                    count += 1
            else:
                BB.extend(dets[iid].tolist())
                image_ids += [iid] * len(dets[iid])
                count += len(dets[iid])

        if BB:
            BB = np.array(BB)
            image_ids = np.array(image_ids)
            confidence = BB[:, -2]
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = image_ids[sorted_ind]
            nd = len(image_ids)

            for d in range(nd):
                R = class_recs[image_ids[d]]
                ovmax = -np.inf
                if R is not None:
                    BBGT = R['bbox'].astype(np.float32)
                    if BBGT.size > 0:
                        bb = BB[d, :].astype(np.float32)

                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(BBGT[:, 0], bb[0])
                        iymin = np.maximum(BBGT[:, 1], bb[1])
                        ixmax = np.minimum(BBGT[:, 2], bb[2])
                        iymax = np.minimum(BBGT[:, 3], bb[3])
                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                               (BBGT[:, 2] - BBGT[:, 0]) *
                               (BBGT[:, 3] - BBGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['det'][jmax]:
                        tp.append(1.)
                        fp.append(0.)
                        R['det'][jmax] = True
                    else:
                        fp.append(1.)
                        tp.append(0.)
                else:
                    fp.append(1.)
                    tp.append(0.)

        assert len(fp) == len(tp) and len(fp) == count
        # compute precision recall
        fp = np.array(fp)
        tp = np.array(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
                
    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t]) 
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) 

    return ap
 

    

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

    all_boxes = {}
    annos = {}
    for index in range(len(test_dataset)):
        image, target = test_dataset.pull_item(index)
        annos[index] = np.array(target) if target else None

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
        all_boxes[index] = dets
        print('post processing time: {:.2f}ms'.format(post_time * 1000))
    do_python_eval(all_boxes, annos, ovthresh=0.5, use_07=True)
    do_python_eval(all_boxes, annos, ovthresh=0.5, use_07=False)
    do_python_eval(all_boxes, annos, ovthresh=0.3, use_07=True)
    do_python_eval(all_boxes, annos, ovthresh=0.3, use_07=False)

