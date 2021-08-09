import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import cv2
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import xBDDatasetTest
from models.hrnet import get_seg_model
from utils import preprocess_inputs, dice


def test(model, data_loader, output_dir):
    dices = []
    _thr = 0.5

    with torch.no_grad():
        for sample in tqdm(data_loader):
            pre_mask = sample['pre_mask'].numpy()
            pre_image = sample['pre_image'].cuda(non_blocking=True)

            out = model(pre_image)

            h, w = pre_mask.shape[1], pre_mask.shape[2]
            out = F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)
            mask_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

            for j in range(mask_pred.shape[0]):
                dices.append(dice(pre_mask[j], mask_pred[j] > _thr))
                
            mask_pred = mask_pred[0] > _thr

            pos = mask_pred == True
            neg = mask_pred == False
            result = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), dtype=np.uint8)
            result[:, :, 0][pos] = 255
            result[:, :, 1][pos] = 255
            result[:, :, 2][pos] = 255
            result[:, :, 0][neg] = 0
            result[:, :, 1][neg] = 0
            result[:, :, 2][neg] = 0

            cv2.imwrite(os.path.join(output_dir, sample['pre_name'][0]), result)
#           plt.imshow(mask_pred)
#           plt.show()
                
    dice_avg = np.mean(dices)
    print('Test Dice: {}'.format(dice_avg))


def main():
    cudnn.benchmark = True

    batch_size = 1

    data_path = 'data/test/'
    output_dir = 'results/test_hrnet_w48_loc/'
    checkpoints_dir = 'checkpoints/'
    snap_to_load = 'hrnet_w48_loc_0.8296'
    cfg_file = 'configs/hrnet_w48_train.yaml'
    with open(cfg_file) as f:
        cfg_str = f.read()
    cfg = yaml.load(cfg_str, Loader=yaml.SafeLoader)

    os.makedirs(output_dir, exist_ok=True)

    model = get_seg_model(cfg).cuda()
    print('loading checkpoint {}'.format(snap_to_load))
    checkpoint = torch.load(os.path.join(checkpoints_dir, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print('loaded checkpoint {} (epoch: {}, best_score: {})'.format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

    model.eval()

    test_dataset = xBDDatasetTest(data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5, shuffle=False, pin_memory=False)

    test(model, test_dataloader, output_dir)


if __name__ == "__main__":
    main()
