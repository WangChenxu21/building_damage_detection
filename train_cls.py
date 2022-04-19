import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tqdm import tqdm
from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import metrics
from logger import create_logger
from utils import AverageMeter
from adamw import AdamW
from dataset import xBDDataset
from builder import build_cls_hierarchy_model, build_cls_model
from losses import FocalLossWithDice
from xview_metric import XviewMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='seresnext', help='seresnext or dpn or resnet or senet')
parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
parser.add_argument('--epoch', type=int, default=10, help='total epoch num')
parser.add_argument('--milestones', nargs='+', default=[1, 3, 5, 7, 9], type=int, help='milestones for reducing lr')
parser.add_argument('--train_batch_size', type=int, default=12, help='batch size for train')
parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for val')
parser.add_argument('--resume', action='store_true', default=False, help='whether to use loc_ckpt')
parser.add_argument('--loc_ckpt', type=str, default=None, help='checkpoint for localization')

## 层次化学习
parser.add_argument('--hierarchy', action='store_true', default=False, help='whether to use hierarchical encoder')
parser.add_argument('--hierarchy_path', type=str, default='hierarchy_data/disaster.taxonomy', help='a file that records parent-child relationship')
parser.add_argument('--hierarchy_prob_json', type=str, default='hierarchy_data/disaster_prob.json', help='a json which records conversion probability between parents and childern')
parser.add_argument('--hierarchical_encoder', type=str, default='GCN', help='TreeLSTM or GCN')
parser.add_argument('--hierarchical_direction', type=str, default='bi_direction', help='bottom_up or top_down or bi_direction')
args = parser.parse_args()

## logger
logger = create_logger(args.model)
configs = 'configs:\n'
for k in list(vars(args).keys()):
    configs += '\t%s: %s\n' % (k, vars(args)[k])
logger.info(configs)
print(configs)

## tensorboard
train_iter = 0
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = f"tensorboard/{args.model}_{args.exp_name}_cls_{rq}"
writer = SummaryWriter(log_dir)


def validate(model, data_loader, epoch, predictions_dir):
    os.makedirs(predictions_dir, exist_ok=True)
    preds_dir = os.path.join(predictions_dir, 'predictions')
    os.makedirs(preds_dir, exist_ok=True)
    targs_dir = os.path.join(predictions_dir, 'targets')
    os.makedirs(targs_dir, exist_ok=True)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            pre_image = sample["pre_image"].cuda(non_blocking=True)
            post_image = sample["post_image"].cuda(non_blocking=True)

            out = model(pre_image, post_image)

            damage_preds = out.cpu().numpy()

            damage_gts = np.array(sample["post_mask"], dtype=np.uint8)
            mask_gts = np.array(sample["pre_mask"], dtype=np.uint8)
            
            for i in range(out.shape[0]):
                damage_pred = damage_preds[i]
                damage_pred = np.argmax(damage_pred, axis=0)

                mask_pred = np.ones(damage_pred.shape)
                mask_pred[damage_pred == 0] = 0

                damage_pred = np.array(damage_pred, dtype=np.uint8)
                mask_pred = np.array(mask_pred, dtype=np.uint8)
                damage_gt = damage_gts[i]
                mask_gt = mask_gts[i]

                cv2.imwrite(os.path.join(preds_dir, "test_localization_" + sample["img_name"][i] + "_prediction.png"), mask_pred)
                cv2.imwrite(os.path.join(preds_dir, "test_damage_" + sample["img_name"][i] + "_prediction.png"), damage_pred)
                cv2.imwrite(os.path.join(targs_dir, "test_localization_" + sample["img_name"][i] + "_target.png"), mask_gt)
                cv2.imwrite(os.path.join(targs_dir, "test_damage_" + sample["img_name"][i] + "_target.png"), damage_gt)

    d = XviewMetrics.compute_score(preds_dir, targs_dir)
    for k, v in d.items():
        print("{}:{}".format(k, v))
        logger.info("{}:{}".format(k, v))
    writer.add_scalar('val/score', d["score"], epoch+1)
    writer.add_scalar('val/damage_f1', d["damage_f1"], epoch+1)
    writer.add_scalar('val/localization_f1', d["localization_f1"], epoch+1)
    writer.add_scalar('val/damage_f1_no_damage', d["damage_f1_no_damage"], epoch+1)
    writer.add_scalar('val/damage_f1_minor_damage', d["damage_f1_minor_damage"], epoch+1)
    writer.add_scalar('val/damage_f1_major_damage', d["damage_f1_major_damage"], epoch+1)
    writer.add_scalar('val/damage_f1_destroyed', d["damage_f1_destroyed"], epoch+1)
    return d["localization_f1"], d["score"]


def evaluate(data_loader, best_score, model, snapshot_name, current_epoch, predictions_dir):
    model = model.eval()

    dice, xview_score = validate(model, data_loader, current_epoch, predictions_dir)
    if xview_score > best_score:
        torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            'dice': dice,
            'xview_score': xview_score,
            }, os.path.join('checkpoints', f'{snapshot_name}_{current_epoch}_{round(xview_score, 3)}')
        )
        best_score = xview_score

    logger.info(f"dice: {dice}, xview_score: {xview_score}, best_score: {best_score}")
    print(f"dice: {dice}, xview_score: {xview_score}, best_score: {best_score}")
    return best_score


def train_epoch(current_epoch, damage_loss, model, optimizer, scheduler, data_loader):
    global train_iter

    losses = AverageMeter()
    ious = AverageMeter()
    f1s = AverageMeter()
    accuracys = AverageMeter()
    recalls = AverageMeter()

    model.train()
    iterator = tqdm(data_loader)
    for sample in iterator:
        pre_image = sample["pre_image"].cuda(non_blocking=True)
        post_image = sample["post_image"].cuda(non_blocking=True)
        post_mask = sample["post_mask"].cuda(non_blocking=True)

        out = model(pre_image, post_image)
        
        ## metrics
        pred = torch.argmax(out, dim=1)
        tp, fp, fn, tn = metrics.get_stats(pred, post_mask, mode='multiclass', num_classes=5)
        iou = metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        ious.update(iou, pre_image.shape[0])
        f1s.update(f1, pre_image.shape[0])
        accuracys.update(accuracy, pre_image.shape[0])
        recalls.update(recall, pre_image.shape[0])

        loss = damage_loss(out, post_mask)
        losses.update(loss.item(), pre_image.size(0))

        writer.add_scalar('train/loss', losses.val, train_iter)
        train_iter += 1

        logger.info(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); IoU {ious.val:.4f} ({ious.avg:.4f}); F1 {f1s.val:.4f} ({f1s.avg:.4f}); Accuracy {accuracys.val:.4f} ({accuracys.avg:.4f}); Recall {recalls.val:.4f} ({recalls.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, ious=ious, f1s=f1s, accuracys=accuracys, recalls=recalls))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); IoU {ious.val:.4f} ({ious.avg:.4f}); F1 {f1s.val:.4f} ({f1s.avg:.4f}); Accuracy {accuracys.val:.4f} ({accuracys.avg:.4f}); Recall {recalls.val:.4f} ({recalls.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, ious=ious, f1s=f1s, accuracys=accuracys, recalls=recalls))

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.1)
        optimizer.step()
        torch.cuda.synchronize()

    scheduler.step(current_epoch)


def main(): 
    global rq

    cudnn.benchmark = True

    batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    snapshot_name = f"{args.model}_{args.exp_name}_cls_{rq}"
    data_path = 'data/train/'
    folds_csv = 'folds.csv'
    predictions_dir = f'predictions_val_{args.model}/'

    np.random.seed(123)
    random.seed(123)
    
    train_dataset = xBDDataset(data_path, 'train', 0, folds_csv) 
    val_dataset = xBDDataset(data_path, 'val', 0, folds_csv)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=5, shuffle=False, pin_memory=False)

    if args.hierarchy:
        model = build_cls_hierarchy_model(args.model, args.hierarchy_path,
                args.hierarchy_prob_json, args.hierarchical_encoder,
                args.hierarchical_direction).cuda()
    else:
        model = build_cls_model(args.model).cuda()

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)

    if args.resume:
        print(f'now loading checkpoint {args.loc_ckpt}...')
        checkpoint = torch.load(args.loc_ckpt, map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
    
    model = model.cuda()

    loss_fn = FocalLossWithDice(5, ce_weight=2, d_weight=0.5, weight=[1, 1, 5, 3, 3]).cuda()

    best_score = 0
    best_score = evaluate(val_data_loader, best_score, model, snapshot_name, -1, predictions_dir)
    torch.cuda.empty_cache()
    for epoch in range(args.epoch):
        train_epoch(epoch, loss_fn, model, optimizer, scheduler, train_data_loader)
        torch.cuda.empty_cache()
        best_score = evaluate(val_data_loader, best_score, model, snapshot_name, epoch, predictions_dir)

    writer.close()


if __name__ == "__main__":
    main()
