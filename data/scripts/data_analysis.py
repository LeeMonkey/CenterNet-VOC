#!/usr/bin/env python
#-*-coding:utf-8-*-

import os
import os.path as osp
import sys
import numpy as np
import warnings
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = list()
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        bbox = []
        for key in ['xmin', 'ymin', 'xmax', 'ymax']:
            value = float(bndbox.findtext(key))
            bbox.append(value)
        assert len(bbox) == 4
        name = obj.findtext('name')
        bbox.append(name)
        boxes.append(bbox)

    return boxes


def analysis(xml_list):
    result = dict()

    for xml in xml_list:
        boxes = parse_xml(xml)
        exists_classes = set()
        for box in boxes:
            w, h = box[2] - box[0], box[3] - box[1]
            aspect_ratio = round(w / h if w / h >= 1 else h / w, 2)
            area = int(w * h)
            name = box[-1]

            if name not in result:
                result[name] = dict()

            if 'area' not in result[name]:
                result[name]['area'] = []

            result[name]['area'].append(area)

            if 'aspect_ratio' not in result[name]:
                result[name]['aspect_ratio'] = []
            result[name]['aspect_ratio'].append(aspect_ratio)

            if 'num_images' not in result[name]:
                result[name]['num_images'] = 0

            if 'num_boxes' not in result[name]:
                result[name]['num_boxes'] = 0
            result[name]['num_boxes']  += 1

            if name not in exists_classes:
                result[name]['num_images'] += 1
            exists_classes.add(name)

    # show result
    areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    areaRngLbl = ['all', 'small', 'medium', 'large']
    num_step = 5
    bpi = dict() 
    for name in result:
        print('{split} {name} {split}'.format(split='='*20, name=name))
        for k in result[name]:
            if k in ['num_images', 'num_boxes']:
                print('{}: {}'.format(k, result[name][k]))

            if k in ['area']:
                print('{}:'.format(k))
                for lbl, re in zip(areaRngLbl, areaRng):
                    left, right = re
                    array = np.array(result[name][k])
                    count = ((array >= left) & (array < right)).sum()
                    print('  {lbl:<6s}: [{l:>6d}x{l:<6d}, {r:>6d}x{r:<6d}]: {c:d}'.format(lbl=lbl,
                        l=int(np.sqrt(left)), r=int(np.sqrt(right)), c=count))

            # aspect_ratio
            if k in ['aspect_ratio']:
                print('{}:'.format(k))
                left = min(result[name][k])
                right = max(result[name][k])
                step = (right - left) / num_step
                array = np.array(result[name][k])
                for i in range(num_step):
                    if i == num_step - 1:
                        count = ((array >= (i * step + left)) & (array <= right)).sum()
                        print('  [{:>9.3f}, {:>9.3f}]: {}'.format(i * step + left, right, count))
                    else:
                        count = ((array >= (i * step + left)) & (array < ((i + 1) * step + left))).sum()
                        print('  [{:>9.3f}, {:>9.3f}): {}'.format(i * step + left, (i + 1) * step + left, count))
        bpi[name] = result[name]['num_boxes'] / result[name]['num_images']
        print('boxes_per_image: {:.2f}'.format(bpi[name]))
        print()

    sum_bpi = sum([bpi[name] for name in bpi])
    print('{split} classes porportion {split}'.format(split='='*20))
    for name in bpi:
        print('{}: {:.2f}'.format(name, sum_bpi / bpi[name]))




def main():
    list_path = sys.argv[1]
    xml_dir = sys.argv[2]
    image_dir = xml_dir.replace('Annotations', 'JPEGImages')
    assert osp.isdir(xml_dir) and osp.isdir(image_dir) 
    exts = ['jpg', 'JPEG', 'png', 'Jpg']

    xml_list = []
    with open(list_path) as f:
        for line in f:
            prefix = line.strip()
            xml_path = osp.join(xml_dir, '{prefix}.xml'.format(prefix=prefix))
            for ext in ['jpg', 'JPEG', 'png', 'Jpg']:
                image_path = osp.join(image_dir, '{prefix:s}.{ext:s}'.format(prefix=prefix, ext=ext))
                if osp.isfile(image_path):
                    if osp.isfile(xml_path) and osp.isfile(image_path):
                        xml_list.append(xml_path)
                    else:
                        warnings.warn('{} without xml file'.format(image_path))
                    break

    analysis(xml_list)
        

if __name__ == '__main__':
    main()
