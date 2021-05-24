import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, target=None):
        return image.astype(np.float32), target

class ToTensor(object):
    def __call__(self, image, target=None):
        image /= 255.
        return image, target


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = image.astype(np.float32)
        mean = np.array(self.mean, dtype=image.dtype)
        std = np.array(self.std, dtype=image.dtype)
        image -= self.mean
        image /= self.std
        return image, target


class Resize(object):
    def __init__(self, size, mean=None):
        self.size = size
        self.mean = mean

    def __call__(self, image, target=None):
        height, width, depth = image.shape
        min_ratio = min(self.size[0] / width, self.size[1] / height)
        resize_image = np.empty((self.size[1], self.size[0], depth), dtype=image.dtype)
        if self.mean is not None:
            resize_image[..., :] = self.mean
        image = cv2.resize(image, None, fx=min_ratio, fy=min_ratio)
        resize_image[:image.shape[0], :image.shape[1]] = image
        if target is not None:
            target[:, :4] *= min_ratio
        return resize_image, target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target=None):
        if random.randrange(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target=None):
        if random.randrange(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target=None):
        if random.randrange(2):
            swap = self.perms[random.randrange(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target=None):
        if random.randrange(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target=None):
        if random.randrange(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, target


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            (-1, -1),
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, target=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            min_iou, max_iou = random.choice(self.sample_options)

            if min_iou == -1 and max_iou == -1:
                return image, target

            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                if target is None:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                                  :]
                    return current_image, target

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(target[:, :4], rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (target[:, :2] + target[:, 2:-1]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = target[mask, :].copy()

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:-1] = np.minimum(current_boxes[:, 2:-1],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:-1] -= rect[:2]

                return current_image, current_boxes


class Expand(object):
    def __init__(self, mean, max_ratio=2):
        self.mean = mean
        self.max_ratio = max_ratio

    def __call__(self, image, target=None):
        if random.randrange(2) or target is None:
            return image, target

        height, width, depth = image.shape
        ratio = random.uniform(1, self.max_ratio)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        target[:, :2] += (int(left), int(top))
        target[:, 2:-1] += (int(left), int(top))

        return image, target


class RandomMirror(object):
    def __call__(self, image, target=None):
        _, width, _ = image.shape
        if random.randrange(2) and target is not None:
            image = image[:, ::-1]
            target = target.copy()
            target[:, 0:-1:2] = width - target[:, 2::-2]
        return image, target


class RandomRotate(object):
    def __init__(self, delta=15, mean=None):
        assert delta >= 0
        assert delta <= 180
        self.delta = delta
        self.mean = mean if mean else (0, 0, 0)

    def __call__(self, image, target=None):
        height, width, depth = image.shape
        center = (width / 2, height / 2)
        angle = random.uniform(-self.delta, self.delta)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nw = int(height * sin + width * cos)
        nh = int(height * cos + width * sin)

        M[0, 2] += (nw / 2) - center[0]
        M[1, 2] += (nh / 2) - center[1]
        image = cv2.warpAffine(image, M, (nw, nh), borderValue=self.mean)

        if target is not None:
            boxes = target[:, :4]
            boxes = np.concatenate((boxes, boxes[:, [0,3,2,1]]), axis=-1).reshape(-1, 2)
            expand_boxes = np.ones((boxes.shape[0], 3), dtype=boxes.dtype)
            expand_boxes[:, :boxes.shape[1]] = boxes
            rotate_boxes = np.dot(M, expand_boxes.T).T
            rotate_boxes = rotate_boxes.reshape(target.shape[0], -1, 2)
            boxes_br = np.squeeze(np.max(rotate_boxes, axis=1))
            boxes_tl = np.squeeze(np.min(rotate_boxes, axis=1))
            target[:, :2] = boxes_tl
            target[:, 2:4] = boxes_br
        
        return image, target


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, target):
        im = image.copy()
        im, target = self.rand_brightness(im, target)
        if random.randrange(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, target = distort(im, target)
        return self.rand_light_noise(im, target)
