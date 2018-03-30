## Python file for joint transformation on input and mask
## From https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask



class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
    
    

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)



class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

    
    
class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))




# class JointRandomSizedCrop(object):
#     '''
#     Random crop the given list of PIL.Image to a random size of (0.08 to 1.0) of the original size
#     and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
#     This is popularly used to train the Inception networks
#     size: size of the smaller edge
#     interpolation: Default: PIL.Image.BILINEAR
#     '''
    
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         self.size = size
#         self.interpolation = interpolation

#     def __call__(self, imgs):
#         for attempt in range(10):
#             area = imgs[0].size[0] * imgs[0].size[1]
#             target_area = random.uniform(0.08, 1.0) * area
#             aspect_ratio = random.uniform(3. / 4, 4. / 3)

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if random.random() < 0.5:
#                 w, h = h, w

#             if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
#                 x1 = random.randint(0, imgs[0].size[0] - w)
#                 y1 = random.randint(0, imgs[0].size[1] - h)

#                 imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
#                 assert(imgs[0].size == (w, h))

#                 return [img.resize((self.size, self.size), self.interpolation) for img in imgs]

#         # Fallback
#         scale = JointScale(self.size, interpolation=self.interpolation)
#         crop = JointCenterCrop(self.size)
#         return crop(scale(imgs))
    
    

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
