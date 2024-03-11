import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

import cv2
import albumentations as A

from utils import bbox_xywh2xyxy


class Transforms:
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.transforms(image=np.array(img))['image']


class BboxClassificationDataset(Dataset):
    def __init__(self, data_paths: list[str], transform=None):
        self.images = []
        self.bboxes = []
        self.labels = []
        for data_path in data_paths:
            data_path = Path(data_path)
            for img_path in (data_path/'images').iterdir():
                label_path = data_path/'labels'/(img_path.stem+'.txt')
                with open(label_path, 'r') as f:
                    lb = f.readlines()
                for bbox in lb:
                    bbox = bbox.split()
                    self.images.append(img_path)
                    self.bboxes.append(list(map(float, bbox[1:])))
                    self.labels.append(int(bbox[0]))
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ih, iw = img.shape[:2]
        xc, yc, bw, bh = self.bboxes[idx]
        xlt, ylt, xrb, yrb = bbox_xywh2xyxy(xc, yc, bw, bh)
        xlt = min(iw-3, int(xlt * iw)) # in order to keep box size > 0
        xrb = max(xlt+3, int(xrb * iw))
        ylt = min(ih-3, int(ylt * ih))
        yrb = max(ylt+3, int(yrb * ih))
        img = img[ylt:yrb, xlt:xrb]
        if 0 in img.shape:
            print(xc, yc, bw, bh)
            print(img.shape)
            print(xlt, ylt, xrb, yrb, iw, ih)

        label = self.labels[idx]

        if self.transform is not None:
            return self.transform(img), label
        return img, label
    
    def get_labels(self):
        return self.labels
    

# class InferDataset(Dataset):
#     def __init__(self, root: str, transform=None):
#         self.root = root
#         self.img_files = sorted(os.listdir(os.path.join(root, 'images')))
        
#         crop_files = sorted(os.listdir(os.path.join(root, 'labels')))
#         self.crops = []
#         for filename in crop_files:
#             with open(os.path.join(root, 'labels', filename), 'r') as crp:
#                 self.crops.append(crp.read().split()[:4])

#         img_files_ = np.array([*map(lambda x: x.split('.')[0], self.img_files)])
#         crop_files_ = np.array([*map(lambda x: x.split('.')[0], crop_files)])
#         assert (img_files_ != crop_files_).sum() == 0, 'Filenames do not match'

#         self.transform = transform

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx):
#         img_name = self.img_files[idx]
#         img_path = os.path.join(self.root, 'images', img_name)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         crop = np.array(self.crops[idx]).astype(np.int32)
#         xlt, ylt, xrb, yrb = crop
#         img = img[ylt:yrb, xlt: xrb]
#         if self.transform is not None:
#             return self.transform(img), crop, img_path
#         return img, crop, img_path


def get_dataset(data_paths: list[str], data_config: dict, pipeline: A.Compose) -> DataLoader:
    transform = Transforms(pipeline)
    dataset = BboxClassificationDataset(
        data_paths,
        transform
    )
    shuffle = data_config['shuffle']
    sampler = None
    if data_config['weighted_sampling']:
        sampler = ImbalancedDatasetSampler(dataset)
        shuffle = None
    loader = DataLoader(dataset,
                        batch_size=data_config['batch_size'],
                        shuffle=shuffle,
                        sampler=sampler,
                        num_workers=data_config['num_workers'],
                        pin_memory=True)
    return loader


# def get_inference_dataset(data: dict, pipeline: dict) -> DataLoader:
#     transform = Transforms(pipeline)
#     dataset = InferDataset(
#         data['root'],
#         transform
#     )
#     loader = DataLoader(dataset,
#                         batch_size=data['batch_size'], 
#                         shuffle=data['shuffle'],
#                         num_workers=data['num_workers'],
#                         pin_memory=True)
#     return loader
