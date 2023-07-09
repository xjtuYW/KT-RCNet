import os
import PIL.Image as Image
import numpy as np
import torch

__all__ = ['CategoryDatasetFolder']


class CategoryDatasetFolder(object):

    def __init__(self, 
                 data_root, 
                 file_name, 
                 transform, 
                 dataset_min_cls: int=0, 
                 dataset_max_cls: int=10000,
                 out_name: bool=False, 
                 DLC_flag: bool=False, 
                 OSS_flag: bool=False):
        data, ori_labels = [], []
        file_name        = file_name + '.csv'
        file_csv         = os.path.join(data_root, file_name)
        with open(file_csv, 'r') as f:  # (name.jpg,label)
            next(f) # skip first line
            for split in f.readlines():
                split = split.strip('\n').split(',')  # split = [name.jpg label]
                if int(split[1]) >= dataset_min_cls and int(split[1]) < dataset_max_cls:
                    data.append(os.path.join(data_root, split[0]))  # name.jpg
                    ori_labels.append(int(split[1]))  # label
        # label_key = sorted(np.unique(np.array(ori_labels)))  # input ['a','a','b','b','c','c'] -> ['a','b','c'] output
        # label_map = dict(zip(label_key, range(len(label_key))))  # input['a','b','c'] - > {'a': 0, 'b': 1, 'c': 2}output
        # mapped_labels = [label_map[x] for x in ori_labels]  # mapped_labels = [0, 0, 1, 1, 2, 2]
        # import pdb
        # pdb.set_trace()
        mapped_labels = ori_labels

        self.DLC        = DLC_flag
        self.OSS        = OSS_flag
        self.file_name  = file_name
        self.data_root  = data_root
        self.transform  = transform
        self.data       = data
        self.labels     = mapped_labels
        self.out_name   = out_name
        self.length     = len(self.data)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        img_path = self.data[index]
        if self.DLC:
            # root/code/fscil/datasets/miniimagenet/images/n0306224500001132.jpg
            img_path = img_path.replace('/root/code/fscil', '/root/data/wy/FSIL')
        if self.OSS:
            oss_cur_dir = os.getcwd()
            img_path = img_path.replace('/mnt/canghe20220320/wy/FSIL', '.')

        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if label > 1000:
            import pdb
            pdb.set_trace()
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label

