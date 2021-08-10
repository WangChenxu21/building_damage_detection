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
from models.hrnet import get_seg_model_double
from losses import dice_round, ComboLoss, FocalLossWithDice


train_iter = 0
log_dir = "tensorboard/hrnet_w48_cls"
writer = SummaryWriter(log_dir)

def validate(model, data_loader, epoch, predictions_dir):
    os.makedirs(predictions_dir, exist_ok=True)
    preds_dir = os.path.join(predictions_dir, 'predictions')
    os.makedirs(preds_dir, exist_ok=True)
    targs_dir = os.path.join(predictions_dir, 'targets')
    os.makedirs(targs_dir, exist_ok=True)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            pre_mask = sample["pre_mask"].numpy()
            post_mask = sample["post_mask"].numpy()
            pre_image = sample["pre_image"].cuda(non_blocking=True)
            post_image = sample["post_image"].cuda(non_blocking=True)

            out = model(pre_image, post_image)

            damage_preds = torch.softmax(output, dim=1).cpu().numpy()
            loc_preds = 1 - torch.sigmoid(out[:, 0, ...])

            for i in range(output.shape[0]):
                damage_pred = damage_preds[i]
                damage_pred = np.argmax(damage_pred, axis=0)
                loc_pred = loc_preds[i]
                loc_pred = (loc_pred[i] > 0.5)
                print('damage_pred:', damage_pred.shape)
                print('loc_pred:', loc_pred.shape)
                assert False

                cv2.imwrite(os.path.join(preds_dir, "test_damage_" + sample["img_name"][i] + "_prediction.png"), argmax)

    dice_avg = np.mean(dices)
    writer.add_scalar('dice/val', dice_avg, epoch)
    print("Val Dice: {}".format(dice_avg))

    return dice_avg


def evaluate(data_loader, best_score, model, snapshot_name, current_epoch, predictions_dir):
    model = model.eval()

    dice, xview_score = validate(model, data_loader, current_epoch, predictions_dir)
    if xview_score > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'dice': dice,
            'xview_score': xview_score,
            }, os.path.join('checkpoints', snapshot_name + '_{}'.format(xview_score)))
        best_score = xview_score

    print('dice: {} xview_score: {} best_score: {}'.format(dice, xview_score, best_score))
    return best_score


def train_epoch(current_epoch, damage_loss, model, optimizer, scheduler, data_loader):
    global train_iter

    losses = AverageMeter()

    model.train()
    iterator = tqdm(data_loader)
    for sample in iterator:
        pre_image = sample["pre_image"].cuda(non_blocking=True)
        post_image = sample["post_image"].cuda(non_blocking=True)
        post_mask = sample["post_mask"].cuda(non_blocking=True)

        out = model(pre_image, post_image)

        loss = damage_loss(out, post_mask)
        losses.update(loss.item(), pre_image.size(0))

        writer.add_scalar('loss/train', losses.val, train_iter)
        train_iter += 1

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses))

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.1)
        optimizer.step()
        torch.cuda.synchronize()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f};".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses))
 

def main(): 
    cudnn.benchmark = True

    batch_size = 1
    val_batch_size = 4

    snapshot_name = "hrnet_w48_cls"
    data_path = 'data/train/'
    folds_csv = 'folds.csv'
    predictions_dir = 'predictions_val/'
    cfg_file = 'configs/hrnet_w48_double_train.yaml'
    with open(cfg_file) as f:
        cfg_str = f.read()
    cfg = yaml.load(cfg_str, Loader=yaml.SafeLoader)

    np.random.seed(123)
    random.seed(123)
    
    train_dataset = xBDDataset(data_path, 'train', 0, folds_csv) 
    val_dataset = xBDDataset(data_path, 'val', 0, folds_csv)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=5, shuffle=False, pin_memory=False)

    model = get_seg_model_double(cfg).cuda()

    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], gamma=0.5)

    loss_fn = FocalLossWithDice(5, ce_weight=2, d_weight=0.5, weight=[1, 1, 5, 3, 3]).cuda()

    best_score = 0
    _cnt = -1
    torch.cuda.empty_cache()
    for epoch in range(16):
        train_epoch(epoch, loss_fn, model, optimizer, scheduler, train_data_loader)
        _cnt += 1
        torch.cuda.empty_cache()
        best_score = evaluate(val_data_loader, best_score, model, snapshot_name, epoch, predictions_dir)

    writer.close()


if __name__ == "__main__":
    main()
