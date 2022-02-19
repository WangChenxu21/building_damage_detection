import os
import re
import random

import cv2
import numpy as np
import pandas as pd
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torch.utils.data import Dataset

from utils import *


class xBDDataset(Dataset):
    def __init__(self, data_path, mode, fold, folds_csv):
        super(Dataset).__init__()

        self.data_path = data_path
        self.mode = mode
        self.fold = fold
        self.folds_csv = folds_csv

        self.df = pd.read_csv(self.folds_csv, dtype={'id': object})
        if self.mode == "train":
            ids = self.df[self.df['fold'] != self.fold]['id'].tolist()
        elif self.mode == "val":
            ids = self.df[self.df['fold'] == self.fold]['id'].tolist()

        self.ids = []
        for id in ids:
            self.ids.append(id + '_1')
            self.ids.append(id + '_2')
            self.ids.append(id + '_3')
            self.ids.append(id + '_4')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pre_img_path = os.path.join(self.data_path, "images", self.ids[idx] + "_pre_disaster.png")
        post_img_path = os.path.join(self.data_path, "images", self.ids[idx] + "_post_disaster.png")
        pre_image = cv2.imread(pre_img_path)
        post_image = cv2.imread(post_img_path)
        
        pre_mask_path = os.path.join(self.data_path, "masks", self.ids[idx] + "_pre_disaster.png")
        post_mask_path = os.path.join(self.data_path, "masks", self.ids[idx] + "_post_disaster.png")
        pre_mask = cv2.imread(pre_mask_path, 0)
        post_mask = cv2.imread(post_mask_path, 0)
        pre_mask[pre_mask == 255] = 1
        post_mask[(pre_mask == 1) & (post_mask == 0)] = 1
        post_mask[post_mask == 255] = 1

        if self.mode == "train":
            input_shape = pre_image.shape[:2]
            # 翻倒
            if random.random() > 0.5:
                pre_image = pre_image[::-1, ...].copy()
                post_image = post_image[::-1, ...].copy()
                pre_mask = pre_mask[::-1, ...].copy()
                post_mask = post_mask[::-1, ...].copy()
            # 翻转
            if random.random() > 0.05:
                rot = random.randrange(4)
                if rot > 0:
                    pre_image = np.rot90(pre_image, k=rot).copy()
                    post_image = np.rot90(post_image, k=rot).copy()
                    pre_mask = np.rot90(pre_mask, k=rot).copy()
                    post_mask = np.rot90(post_mask, k=rot).copy()
            # 平移
            if random.random() > 0.9:
                shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
                pre_image = shift_image(pre_image, shift_pnt)
                post_image = shift_image(post_image, shift_pnt)
                pre_mask = shift_image(pre_mask, shift_pnt)
                post_mask = shift_image(post_mask, shift_pnt)
            # 旋转缩放
            if random.random() > 0.9:
                rot_pnt = (pre_image.shape[0] // 2 + random.randint(-320, 320), pre_image.shape[1] // 2 + random.randint(-320, 320))
                scale = 0.9 + random.random() * 0.2
                angle = random.randint(0, 20) - 10
                if (angle != 0) or (scale != 1):
                    pre_image = rotate_image(pre_image, angle, scale, rot_pnt)
                    post_image = rotate_image(post_image, angle, scale, rot_pnt)
                    pre_mask = rotate_image(pre_mask, angle, scale, rot_pnt)
                    post_mask = rotate_image(post_mask, angle, scale, rot_pnt)
            # 通道偏移
            if random.random() > 0.99:
                pre_image = shift_channels(pre_image, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
                post_image = shift_channels(post_image, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            # hsv
            if random.random() > 0.99:
                pre_image = change_hsv(pre_image, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
                post_image = change_hsv(post_image, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            # 亮度
            if random.random() > 0.99:
                if random.random() > 0.99:
                    pre_image = clahe(pre_image)
                    post_image = clahe(post_image)
                elif random.random() > 0.99:
                    pre_image = gauss_noise(pre_image)
                    post_image = gauss_noise(post_image)
                elif random.random() > 0.99:
                    pre_image = cv2.blur(pre_image, (3, 3))
                    post_image = cv2.blur(post_image, (3, 3))
            elif random.random() > 0.99:
                if random.random() > 0.99:
                    pre_image = saturation(pre_image, 0.9 + random.random() * 0.2)
                    post_image = saturation(post_image, 0.9 + random.random() * 0.2)
                elif random.random() > 0.99:
                    pre_image = brightness(pre_image, 0.9 + random.random() * 0.2)
                    post_image = brightness(post_image, 0.9 + random.random() * 0.2)
                elif random.random() > 0.99:
                    pre_image = contrast(pre_image, 0.9 + random.random() * 0.2)
                    post_image = contrast(post_image, 0.9 + random.random() * 0.2)

        pre_image = preprocess_inputs(pre_image)
        post_image = preprocess_inputs(post_image)

        pre_image = torch.from_numpy(pre_image.transpose((2, 0, 1))).float()
        post_image = torch.from_numpy(post_image.transpose((2, 0, 1))).float()
        pre_mask = torch.from_numpy(pre_mask).long()
        post_mask = torch.from_numpy(post_mask).long()

        sample = {
                'pre_image': pre_image,
                'post_image': post_image,
                'pre_mask': pre_mask,
                'post_mask': post_mask,
                'pre_name': self.ids[idx] + "_pre_disaster.png",
                'post_name': self.ids[idx] + "_post_disaster.png",
                'img_name': self.ids[idx],
                }
        return sample


class xBDDatasetTest(Dataset):
    def __init__(self, data_path):
        super(Dataset).__init__()

        self.data_path = data_path

        ids = set()
        for filename in os.listdir(os.path.join(self.data_path, 'images')):
            f = re.sub('_pre_disaster.png', '', filename)
            f = re.sub('_post_disaster.png', '', f)
            ids.add(f)
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pre_img_path = os.path.join(self.data_path, "images", self.ids[idx] + "_pre_disaster.png")
        post_img_path = os.path.join(self.data_path, "images", self.ids[idx] + "_post_disaster.png")
        pre_image = cv2.imread(pre_img_path)
        post_image = cv2.imread(post_img_path)

        pre_mask_path = os.path.join(self.data_path, "masks", self.ids[idx] + "_pre_disaster.png")
        post_mask_path = os.path.join(self.data_path, "masks",  self.ids[idx] + "_post_disaster.png")
        pre_mask = cv2.imread(pre_mask_path, 0)
        post_mask = cv2.imread(post_mask_path, 0)
        pre_mask[pre_mask == 255] = 1
        post_mask[(pre_mask == 1) & (post_mask == 0)] = 1
        post_mask[post_mask == 255] = 1

        pre_image = preprocess_inputs(pre_image)
        post_image = preprocess_inputs(post_image)

        pre_image = torch.from_numpy(pre_image.transpose((2, 0, 1))).float()
        post_image = torch.from_numpy(post_image.transpose((2, 0, 1))).float()
        pre_mask = torch.from_numpy(pre_mask).long()
        post_mask = torch.from_numpy(post_mask).long()

        sample = {
                'pre_image': pre_image,
                'post_image': post_image,
                'pre_mask': pre_mask,
                'post_mask': post_mask,
                'pre_name': self.ids[idx] + "_pre_disaster.png",
                'post_name': self.ids[idx] + "_post_disaster.png",
                'img_name': self.ids[idx],
                }
        return sample


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    data_path = 'data/train/'
    test_data_path = 'data/test/'
    fold = 0
    folds_csv = 'folds.csv'

    train_dataset = xBDDataset(data_path, "train", fold, folds_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = xBDDatasetTest(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for sample in train_dataloader:
        print(sample.keys())
        print(sample['img_name'])
        print('pre_image:', sample['pre_image'].shape)
        print('post_image:', sample['post_image'].shape)
        print('pre_mask:', sample['pre_mask'].shape, np.unique(np.array(sample['pre_mask'])))
        print('post_mask:', sample['post_mask'].shape, np.unique(np.array(sample['post_mask'])))
        for idx in range(sample['pre_image'].shape[0]):
            ax = plt.subplot(221)
            ax.set_title('pre_image')
            pre_image = (sample['pre_image'][idx].permute(1, 2, 0) + 1) * 127
            plt.imshow(pre_image.int())
            ax = plt.subplot(222)
            ax.set_title('post_image')
            post_image = (sample['post_image'][idx].permute(1, 2, 0) + 1) * 127
            plt.imshow(post_image.int())
            ax = plt.subplot(223)
            ax.set_title('pre_mask')
            plt.imshow(sample['pre_mask'][idx])
            ax = plt.subplot(224)
            ax.set_title('post_mask')
            plt.imshow(sample['post_mask'][idx])
            plt.show()
