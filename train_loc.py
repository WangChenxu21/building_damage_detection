import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random

import yaml
import numpy as np
from tqdm import tqdm
from apex import amp

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import dice, AverageMeter
from adamw import AdamW
from dataset import xBDDataset
from models.hrnet import get_seg_model
from losses import dice_round, ComboLoss


train_iter = 0
log_dir = "tensorboard/hrnet_w48_test"
writer = SummaryWriter(log_dir)

def validate(model, data_loader, epoch):
    dices = []
    _thr = 0.5

    with torch.no_grad():
        for sample in tqdm(data_loader):
            pre_mask = sample["pre_mask"].numpy()
            pre_image = sample["pre_image"].cuda(non_blocking=True)

            out = model(pre_image)

            h, w = pre_mask.shape[1], pre_mask.shape[2]
            out = F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)
            mask_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

            for j in range(mask_pred.shape[0]):
                dices.append(dice(pre_mask[j], mask_pred[j] > _thr))

    dice_avg = np.mean(dices)
    writer.add_scalar('dice/val', dice_avg, epoch)
    print("Val Dice: {}".format(dice_avg))

    return dice_avg


def evaluate(data_loader, best_score, model, snapshot_name, current_epoch):
    model = model.eval()

    d = validate(model, data_loader, current_epoch)
    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
            }, os.path.join('checkpoints', snapshot_name + '_{}'.format(d)))
        best_score = d

    print('score: {} best_score: {}'.format(d, best_score))
    return best_score


def train_epoch(current_epoch, seg_loss, model, optimizer, scheduler, data_loader):
    global train_iter

    losses = AverageMeter()
    dices = AverageMeter()

    model.train()
    iterator = tqdm(data_loader)
    for sample in iterator:
        pre_image = sample["pre_image"].cuda(non_blocking=True)
        pre_mask = sample["pre_mask"].cuda(non_blocking=True)

        out = model(pre_image)

        ph, pw = out.size(2), out.size(3)
        h, w = pre_mask.size(1), pre_mask.size(2)
        if ph != h or pw != w:
            out = F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)
        loss = seg_loss(out, pre_mask)

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
#            dice_sc = 1 - dice_round(_probs, pre_mask[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, pre_mask)


        losses.update(loss.item(), pre_image.shape[0])
        dices.update(dice_sc, pre_image.shape[0])

        writer.add_scalar('loss/train', losses.val, train_iter)
        writer.add_scalar('dice/train', dices.val, train_iter)
        train_iter += 1

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.1)
        optimizer.step()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Dice {dice.avg:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))
 

def main(): 
    cudnn.benchmark = True

    batch_size = 2
    val_batch_size = 4

    snapshot_name = "hrnet_w48_loc"
    data_path = 'data/train/'
    folds_csv = 'folds.csv'
    cfg_file = 'configs/hrnet_w48_train.yaml'
    with open(cfg_file) as f:
        cfg_str = f.read()
    cfg = yaml.load(cfg_str, Loader=yaml.SafeLoader)

    np.random.seed(123)
    random.seed(123)
    
    train_dataset = xBDDataset(data_path, 'train', 0, folds_csv) 
    val_dataset = xBDDataset(data_path, 'val', 0, folds_csv)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=5, shuffle=False, pin_memory=False)

    model = get_seg_model(cfg).cuda()

    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], gamma=0.5)

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()

    best_score = 0
    _cnt = -1
    torch.cuda.empty_cache()
    for epoch in range(16):
        train_epoch(epoch, seg_loss, model, optimizer, scheduler, train_data_loader)
        _cnt += 1
        torch.cuda.empty_cache()
        best_score = evaluate(val_data_loader, best_score, model, snapshot_name, epoch)

    writer.close()


if __name__ == "__main__":
    main()
