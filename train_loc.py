import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from apex import amp

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import dice, iou, AverageMeter
from adamw import AdamW
from dataset import xBDDataset
from builder import build_loc_model
from losses import dice_round, ComboLoss


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='seresnext50 or dpn92 or res34 or senet154')
parser.add_argument('--epoch', type=int, help='total epoch num')
parser.add_argument('--train_batch_size', type=int, help='batch size for train')
parser.add_argument('--val_batch_size', type=int, help='batch size for val')
args = parser.parse_args()

## tensorboard
train_iter = 0
log_dir = f"tensorboard/{args.model}_loc"
writer = SummaryWriter(log_dir)

def validate(model, data_loader, epoch):
    dices = []
    ious = []
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
                ious.append(iou(pre_mask[j], mask_pred[j] > _thr))

    dice_avg = np.mean(dices)
    iou_avg = np.mean(ious)
    writer.add_scalar('dice/val', dice_avg, epoch+1)
    writer.add_scalar('iou/val', iou_avg, epoch+1)
    print(f"Val Dice: {dice_avg}, Val IoU: {iou_avg}")

    return dice_avg, iou_avg


def evaluate(data_loader, best_score, model, snapshot_name, current_epoch):
    model = model.eval()

    dice, iou = validate(model, data_loader, current_epoch)
    if dice > best_score:
        torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            'best_score': dice,
            }, os.path.join('checkpoints', f'{snapshot_name}_{current_epoch}_{round(dice, 3)}')
        )
        best_score = dice

    print(f"dice: {dice}, iou: {iou}, best_score: {best_score}")
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

    batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    snapshot_name = f"{args.model}_loc"
    data_path = 'data/train/'
    folds_csv = 'folds.csv'

    np.random.seed(123)
    random.seed(123)
    
    train_dataset = xBDDataset(data_path, 'train', 0, folds_csv) 
    val_dataset = xBDDataset(data_path, 'val', 0, folds_csv)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=5, shuffle=False, pin_memory=False)

    model = build_loc_model(args.model).cuda()

    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()

    best_score = evaluate(val_data_loader, 0, model, snapshot_name, -1)
    torch.cuda.empty_cache()
    for epoch in range(args.epoch):
        train_epoch(epoch, seg_loss, model, optimizer, scheduler, train_data_loader)
        torch.cuda.empty_cache()
        best_score = evaluate(val_data_loader, best_score, model, snapshot_name, epoch)

    writer.close()


if __name__ == "__main__":
    main()
