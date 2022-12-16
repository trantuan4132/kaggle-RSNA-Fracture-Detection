import os, glob, argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(
#             inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss


class WeightedLoss(nn.Module):
    def __init__(self, loss_fn='bce', weights=[2, 1], reduction='mean'):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none') if loss_fn=='bce' else nn.MSELoss(reduction='none')
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = self.loss_fn(inputs, targets)
        weights = (targets!=0) * self.weights[0] + (targets==0) * self.weights[1]
        loss = (loss * weights).sum(axis=1) / weights.sum(axis=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BoxLoss(nn.Module):
    def __init__(self, bb_loss_fn=nn.L1Loss(), 
                 cls_loss_fn=nn.BCEWithLogitsLoss(), 
                 weights=[0.8, 0.2]):
        super().__init__()
        self.bb_loss_fn = bb_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.weights = weights

    def forward(self, inputs, targets):
        pos_idx = (targets[..., 4:] > 0.5).any(axis=-1)
        bb_loss = self.bb_loss_fn(inputs[pos_idx, :4], targets[pos_idx, :4]) \
                  if pos_idx.sum().item() > 0 else 0
        cls_loss = self.cls_loss_fn(inputs[..., 4:], targets[..., 4:]) 
        loss = self.weights[0] * bb_loss + self.weights[1] * cls_loss
        return loss


def train_one_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    running_loss = AverageMeter()
    tepoch = tqdm(loader)
    # with tqdm(total=len(loader)) as tepoch:
    for batch_idx, (data, targets) in enumerate(tepoch):
        data = data.to(config.device) 
        targets = targets.float().to(config.device)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # Update progress bar
        running_loss.update(loss.item(), data.size(0))
        # tepoch.set_postfix(loss=running_loss.avg)
        tepoch.set_description_str(f'Loss: {running_loss.avg:.4f}')

    return running_loss.avg


def valid_one_epoch(model, loader, criterion, config):
    model.eval()
    running_loss = AverageMeter()
    preds = []
    gts = []
    # with tqdm(total=len(loader)) as tepoch:
    tepoch = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tepoch):
            data, targets = data.to(config.device), targets.to(config.device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            preds.append(outputs.cpu())
            gts.append(targets.cpu())

            # Update progress bar
            running_loss.update(loss.item(), data.size(0))
            # tepoch.set_postfix(loss=running_loss.avg)
            tepoch.set_description_str(f'Loss: {running_loss.avg:.4f}')

    # Calculate accuracy and AUC score
    metrics = {}
    if len(config._metric_to_use) > 0:
        preds = torch.cat(preds).sigmoid().cpu().detach().numpy()
        gts = torch.cat(gts).cpu().detach().numpy()
        if 'acc' in config._metric_to_use:
            metrics['acc'] = np.mean(np.mean((preds > 0.5) == gts, axis=0))
        if 'auc' in config._metric_to_use:
            auc_scores = {}
            for i, col in enumerate(config.label_cols[4:] if config.bbox_label else config.label_cols):
                auc_scores[col] = roc_auc_score(gts[:, i], preds[:, i])
            metrics['auc'] = np.mean(list(auc_scores.values()))
    return running_loss.avg, metrics


def run(fold, config):
    # Prepare train and val set
    full_train_df = pd.read_pickle(config.label_file) \
                    if config.label_file.endswith('.pkl') \
                    else pd.read_csv(config.label_file)
    train_df = full_train_df.query(f"fold!={fold}")
    val_df = full_train_df.query(f"fold=={fold}")

    if config.debug:
        train_df = train_df.sample(80)
        val_df = val_df.sample(640)

    kwargs = dict(additional_targets=dict((f'image{i}', 'image') for i in range(config.seq_len-1 if config.seq_len else 0)))
    if config.bbox_label: 
        kwargs['bbox_params'] = A.BboxParams(format='albumentations', label_fields=['class_labels'])

    train_transform = build_transform(config.image_size, is_train=True, include_top=True, **kwargs)
    val_transform = build_transform(config.image_size, is_train=False, include_top=True, **kwargs)
    
    train_dataset = RSNAClassificationDataset(image_dir=config.image_dir, df=train_df, 
                                              img_cols=config.img_cols, label_cols=config.label_cols,
                                              img_format=config.img_format, bbox_label=config.bbox_label, 
                                              transform=train_transform, use_2dot5D=config.use_2dot5D, 
                                              overlap=True, seq_len=config.seq_len, crop_cols=config.crop_cols)
    val_dataset = RSNAClassificationDataset(image_dir=config.image_dir, df=val_df,
                                            img_cols=config.img_cols, label_cols=config.label_cols,
                                            img_format=config.img_format, bbox_label=config.bbox_label, 
                                            transform=val_transform, use_2dot5D=config.use_2dot5D, 
                                            overlap=True, seq_len=config.seq_len, crop_cols=config.crop_cols)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    
    # Train on full data
    if len(val_dataset) == 0: 
        config._metric_to_opt = 'loss'
        config._metric_to_use = []

    # Initialize model
    model = RSNAClassifier(config.model_name, pretrained=config.pretrained,
                           checkpoint_path=config.checkpoint_path, 
                           in_chans=config.in_chans, num_classes=config.num_classes,
                           drop_path_rate=config.drop_path_rate, use_seq_layer=config.use_seq_layer)
    model = model.to(config.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    # print(f"LR: {config.learning_rate}")

    # Set up training
    start_epoch = 0
    criterion_lst = {
        'bce': nn.BCEWithLogitsLoss(),
        'weighted_bce': WeightedLoss(loss_fn='bce', weights=config.weights),
        'weighted_mse': WeightedLoss(loss_fn='mse', weights=config.weights),
        'box': BoxLoss(weights=config.weights),
        'box_ratio': BoxLoss(weights=config.weights, cls_loss_fn=nn.L1Loss()),
    }
    criterion = criterion_lst[config.loss_fn]
    optimizer = eval(f"optim.{config.optimizer}(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)")
    scheduler = eval(f"optim.lr_scheduler.{config.lr_scheduler}(optimizer, **config.lr_scheduler_params)")
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter() if config._use_tensorboard else None

    # Load checkpoint
    if config.resume:
        checkpoint = load_checkpoint(fold=fold, checkpoint_dir=config.checkpoint_dir)
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
    print(f"LR: {scheduler.get_last_lr()[0]}")

    # Initialize wandb logging if set to True
    if config._use_wandb:
        wandb.init(project=config._wandb_project, name=f'{config.model_name}-{config.image_size}-fold{fold}',
                   entity=config._wandb_entity, config={k: v for k, v in vars(config).items() if not k.startswith('_')},
                   id=config._wandb_resume_id, resume='must' if config._wandb_resume_id else None)
        if len(val_dataset) > 0: wandb.define_metric("val/loss", summary="min")
        for metric in config._metric_to_use:
            wandb.define_metric(f"val/{metric}", summary="max")

    best_score = np.inf if config._metric_to_opt == 'loss' else 0
    for epoch in range(start_epoch, config.n_epochs):
        print(f'Fold: {fold}  Epoch: {epoch}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config)
        val_loss, metrics = valid_one_epoch(model, val_loader, criterion, config)
        scheduler.step()
        print(', '.join([f"{metric.upper()}: {metrics[metric]:.4f}" for metric in config._metric_to_use]))
        print(f'New LR: {scheduler.get_last_lr()[0]}')

        # Log to file
        log_stats = {
            'fold': fold, 'epoch': epoch, 'n_parameters': n_parameters,
            'train_loss': train_loss,
        }
        if len(val_dataset) > 0: log_stats['val_loss'] = val_loss
        for metric in config._metric_to_use:
            log_stats['val_' + metric] = metrics[metric]
        log_to_file(log_stats, "log.txt", config.checkpoint_dir)

        # Wandb logging
        log_dict = {'train/loss': train_loss}
        if len(val_dataset) > 0: log_dict['val/loss'] = val_loss
        for metric in config._metric_to_use:
            log_dict['val/' + metric] = metrics[metric]
        if config._use_wandb:
            wandb.log(log_dict)

        # Tensorboard logging
        if writer:
            for k, v in log_dict.items():
                writer.add_scalar(k, v, global_step=epoch)

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
        }
        score = log_stats[('val_' if len(val_dataset) > 0 else 'train_') + config._metric_to_opt]
        save_path = f"{config.checkpoint_dir}/fold={fold}-epoch={epoch}-{config._metric_to_opt}={score:.4f}.pth"
        save_checkpoint(checkpoint, save_path)
        if (score > best_score and config._metric_to_opt != 'loss') or \
           (score < best_score and config._metric_to_opt == 'loss'):
            best_score = score
            checkpoint[config._metric_to_opt] = score
            save_path = f"{config.checkpoint_dir}/fold={fold}-best.pth"
            print('--> Saving checkpoint')
            save_checkpoint(checkpoint, save_path)

        # Delete old checkpoint
        for fname in glob.glob(f"{config.checkpoint_dir}/fold={fold}-epoch={epoch-config.save_freq}*"):
            os.remove(fname)

    # Finish wandb logging
    if config._use_wandb:
        # wandb.save(f"{config.checkpoint_dir}/fold={fold}-best.pth")
        wandb.finish()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--CFG', type=str, default='config/CFG_FD_train.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--fold', type=str, default='all', 
                        help="0 ⟶ (kfold-1): train 1 fold only, 'all': train all folds sequentially")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.CFG, mode='r') as stream:
        config = Struct(**yaml.safe_load(stream))
    set_seed(config.seed)
    if os.path.exists('/kaggle/input'):
        config.image_dir = '../input/rsna-2022-cervical-spine-fracture-detection/' + os.path.basename(config.image_dir)
    if os.path.exists('/content/drive/MyDrive') and not config.checkpoint_dir.startswith(('/content/drive', 'drive')):
        config.checkpoint_dir = os.path.join('/content/drive/MyDrive/rsna-checkpoint', config.checkpoint_dir, 
                                            f'{config.model_name}-{config.image_size}')
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.num_classes = len(config.label_cols)

    # Train model
    if args.fold == 'all':
        for fold in range(config.kfold):
            run(fold, config)
    else:
        config.fold = int(args.fold)
        run(config.fold, config)

if __name__ == "__main__":
    main()