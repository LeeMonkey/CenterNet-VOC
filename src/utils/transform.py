#!/usr/bin/env python
#-*-coding:utf-8-*-
import src.config as config
from .augmentations import Compose, ConvertFromInts, \
    PhotometricDistort, Expand, RandomSampleCrop, \
    RandomMirror, RandomRotate, Resize, Normalize
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

class CenterNetTransform(object):
    def __init__(self, size=[640], mean=(104, 117, 123), std=None, is_training=True):
        self.mean = mean
        self.std = std if std else (1, 1, 1)
        assert len(self.mean) == 3
        assert len(self.std) == len(self.mean)

        assert isinstance(size, (list, tuple)) or len(size) >= 1
        if len(size) == 1:
            self.size = size * 2
        else:
            self.size = size[:2]
        if is_training:
            self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(self.mean),
                RandomSampleCrop(),
                RandomMirror(),
                RandomRotate(mean=self.mean),
                Resize(self.size, self.mean),
                Normalize(self.mean, self.std)
                #SubtractMeans(self.mean)
            ])
        else:
            self.augment = Compose([
                ConvertFromInts(),
                Resize(self.size, self.mean),
                Normalize(self.mean, self.std)
            ])

    def __call__(self, img, target=None):
        return self.augment(img, target)


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or \
            dict(zip(config.CENTERNET_CLASSES, range(len(config.CENTERNET_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = None
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if not self.keep_difficult:
                if difficult is not None and int(difficult.text) == 1:
                    continue
            name = obj.findtext('name').lower().strip()
            bndbox = obj.find('bndbox')
            if labels is None:
                labels = list()
            tags = ['xmin', 'ymin', 'xmax', 'ymax']
            labels.append([float(bndbox.findtext(t)) - 1 for t in tags])
            labels[-1].append(self.class_to_ind[name])
            assert len(labels[-1]) == 5
        return labels


if __name__ == '__main__':
    aug = CenterNetAugmentation()
    image = cv2.imread('/home/seeta-docods/Workspace/framework/'
        'Object-Detection/CenterNet-Armour/data/Armour/JPEGImages/DJI_0104_1.jpg')
    target = np.array([[223, 189, 277, 221, 1]], dtype=np.float32)
    image, target = aug(image, target)

    for box in target:
        cv2.rectangle(image, 
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 0, 255),
                      2)
    cv2.imwrite('1.jpg', image)
