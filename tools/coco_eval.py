# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:19:02 2020

@author: Lee
"""
import os
import sys
import cv2
import math
import time
import torch
import numpy as np
sys.path.append("./backbone")
from mobilenetv2 import MobileNet
from dataset import ctDataset
from predict import pre_process, ctdet_decode, post_process, merge_outputs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =============================================================================
# 推断
# =============================================================================
def process(images, return_time=False):
    with torch.no_grad():
      output = model(images)
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] 
     
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, K=100) # K 是最多保留几个目标
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

# =============================================================================
# 常规 IOU
# =============================================================================
def bbox_iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = bbox1[0] - bbox1[2] / 2.0, bbox1[1] - bbox1[3] / 2.0
        xmax1, ymax1 = bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0
        xmin2, ymin2 = bbox2[0] - bbox2[2] / 2.0, bbox2[1] - bbox2[3] / 2.0
        xmax2, ymax2 = bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1 ) 
    area2 = (xmax2 - xmin2 ) * (ymax2 - ymin2 )
 
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou
#bbox1 = [1,1,2,2]
#bbox2 = [2,2,2,2]
#ret = iou(bbox1,bbox2,True)
    


# =============================================================================
# 旋转 IOU
# =============================================================================
def iou_rotate_calculate(boxes1, boxes2):
#    print("####boxes2:", boxes1.shape)
#    print("####boxes2:", boxes2.shape)
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
#        print(int_area)
    else:
        ious=0
    return ious
# 用中心点坐标、长宽、旋转角
#boxes1 = np.array([1,1,2,2,0],dtype='float32')
#boxes2 = np.array([2,2,2,2,0],dtype='float32')
#ret = iou_rotate_calculate(boxes1,boxes2)
    


# =============================================================================
# 获得标签信息
# =============================================================================
def get_lab_ret(xml_path):    
    ret = []
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                x1 = float(ob[0])
                y1 = float(ob[1])
                w = float(ob[2])
                h = float(ob[3])
                bbox = [x1, y1, w, h]  # COCO 对应格式[x,y,w,h]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret
    

def get_pre_ret(img_path, device, hm_thresh=0.1):
    image = cv2.imread(img_path)
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)
    
    dets = post_process(dets, meta)
    ret = merge_outputs(dets)
    
    res = np.empty([1, 6])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:,4] > hm_thresh]
        tmp_c = np.ones(len(tmp_s)) * i
        tmp = np.c_[tmp_c,tmp_s]
        res = np.append(res,tmp,axis=0)
    res = np.delete(res, 0, 0)
    #res = res.tolist()
    return res


def py_cpu_nms(all_dets, thresh=0.45, global_nms=True):
    """Pure Python NMS baseline."""
    nms_dets = []
    if global_nms:
        dets = all_dets
        x1 = dets[:, 1]
        y1 = dets[:, 2]
        x2 = dets[:, 3]
        y2 = dets[:, 4]
        scores = dets[:, 5]

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
            dets = all_dets[all_dets[:, 0] == cls]
            x1 = dets[:, 1]
            y1 = dets[:, 2]
            x2 = dets[:, 3]
            y2 = dets[:, 4]
            scores = dets[:, 5]

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


def eval(image_dir, test_json_path, device, hm_thresh=0.25, nms_thresh=0.3):
    assert os.path.isdir(image_dir)
    coco = COCO(test_json_path)
    img_ids = coco.getImgIds()

    pred_labels = []

    for i, iid in enumerate(img_ids):
        anno = coco.loadImgs(iid)[0]
        image_path = os.path.join(image_dir, anno["file_name"])

        torch.cuda.synchronize()
        st = time.time()

        pred_results = get_pre_ret(image_path, device, hm_thresh) 
        pred_results = py_cpu_nms(pred_results, nms_thresh, global_nms=True)

        torch.cuda.synchronize()
        et = time.time()
        print("No.{:d} time: {:.2f}ms".format(i + 1, (et - st) * 1000))
        print("-" * 50)
        for cls, xmin, ymin, xmax, ymax, score in pred_results:
            pred_labels.append({
                "image_id": iid,
                "category_id": cls,
                "bbox": [
                    int(xmin), 
                    int(ymin), 
                    int(xmax - xmin), 
                    int(ymax - ymin)],
                "score":float(score)
            })

    if not os.path.exists("./labels"):
        os.makedirs("./labels")

    save_path = os.path.join("./labels", "preds.json")
    import json
    with open(save_path, "w") as f:
        json.dump(pred_labels, f, indent=2)
        
    coco_dets = coco.loadRes(save_path)
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_path = "models/Armour/20210519/640x640/last.pth"
    image_dir = "data/Armour/JPEGImages"
    test_json_path = "data/Armour/coco_test.json"
    infrared_json_path = "data/Armour/infrared_coco_test.json"
    model = MobileNet(num_classes=ctDataset.num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    with torch.no_grad():
        eval(image_dir, test_json_path, device)
        eval(image_dir, infrared_json_path, device)
