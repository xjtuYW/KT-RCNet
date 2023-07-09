from __future__ import absolute_import
from __future__ import division
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import math
from .autoaugment import CIFAR10Policy, Cutout, AutoAugImageNetPolicy


def get_transforms(image_size):
    transforms_train_inc = {
        'cifar100': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
                ),

        'miniImageNet': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
        ),

        'ImageNet_R': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
        ),

        'ImageNet': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
        ),

        'cub_200': transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        ),
    }

    if image_size == 224:
        transforms_test_inc = {
            'cifar100': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            ),

            'miniImageNet': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'ImageNet_R': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'ImageNet': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'cub_200': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ),
        }
    else:
        
        transforms_test_inc = {
            'cifar100': transforms.Compose([
                transforms.Resize(336),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            ),
            'miniImageNet': transforms.Compose([
                transforms.Resize(336),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'ImageNet_R': transforms.Compose([
                transforms.Resize(336),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'ImageNet': transforms.Compose([
                transforms.Resize(336),
                transforms.ToTensor(),
                transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
            ),

            'cub_200': transforms.Compose([
                transforms.Resize(336),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ),
        }
    return transforms_train_inc, transforms_test_inc

def image_augment(state='train', dataset='miniImageNet',image_size=224):
    """
    @state: in which stage, e.g., training stage, validation stage, or testing stage
    @dataset: currently used dataset
    """
    transforms_train_inc, transforms_test_inc = get_transforms(image_size)
    if state == 'train':
        return transforms_train_inc[dataset]
    else:
        return transforms_test_inc[dataset]

