import os
import PIL.Image as Image
import numpy as np
import torch
from .bucket import *

__all__ = ['Datasets']


class Datasets(object):

    def __init__(self, img_root, files, transform, data_in_oss=False, bucket_name=None, retries=10, out_name=False): 
        """
        get datasets

        @para img_root: the root dir of image
        @para files: list, files which save the name and the label of image, jpg_name, label
        @para transform: image augmentation
        @para out_name: if True, return the name of jpg.

        """
        self.img_root = img_root
        self.files = files
        self.transform = transform
        self.imgs, self.labels = [], []
        for fi in self.files:
            with open(fi) as f:
                next(f)
                for jpg_label in f.readlines():
                    jpg, label = jpg_label.strip('\n').split(',')
                    self.imgs.append(jpg)
                    self.labels.append(int(label))
        
        label_key = sorted(np.unique(np.array(self.labels)))  # input ['a','a','b','b','c','c'] -> ['a','b','c'] output
        label_map = dict(zip(label_key, range(len(label_key))))  # input['a','b','c'] - > {'a': 0, 'b': 1, 'c': 2}output
        mapped_labels = [label_map[x] for x in self.labels]  # mapped_labels = [0, 0, 1, 1, 2, 2]
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.imgs)
        self.data_in_oss = data_in_oss

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.imgs[index])
        if self.data_in_oss:
            img_path = read_from_buffer(self.data_bucket, img_path, self.bucket, retries=self.retries)

        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label

