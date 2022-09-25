import os
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import RSNAClassificationDataset, build_transform
from model import RSNAClassifier
from utils import load_checkpoint

class CFG_CSC:
    # Data
    input_dir = '.'
    label_file = 'train.csv'
    image_folder = 'train_images'
    img_cols = ['StudyInstanceUID', 'Slice']
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    batch_size = 32
    image_size = 512
    num_workers = 2
    pin_memory = True
    seed = 42
    out_file = 'train_CSC_full.csv'
    mode = 'a'                              # 'w': write to file, 'a': append to file

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_chans = 3
    num_classes = len(label_cols)
    drop_path_rate = 0.1
    pretrained = False                       # True: load pretrained model, False: train from scratch
    checkpoint_path = ''                    # Path to model's pretrained weights
    CSC_checkpoint_dirs = {'convnext_tiny': ['CSC_checkpoint/convnext_tiny/fold=0-best.pth',
                                             'CSC_checkpoint/convnext_tiny/fold=1-best.pth',
                                             'CSC_checkpoint/convnext_tiny/fold=2-best.pth',
                                             'CSC_checkpoint/convnext_tiny/fold=3-best.pth',
                                             'CSC_checkpoint/convnext_tiny/fold=4-best.pth']}
    debug = False


class CFG_CSC_FD:
    label_file = 'sample_submission.csv'
    image_folder = 'test_images'
    out_file = 'submission.csv'
    FD_checkpoint_dirs = {'convnext_tiny': ['FD_checkpoint/convnext_tiny/fold=0-best.pth',
                                            'FD_checkpoint/convnext_tiny/fold=1-best.pth',
                                            'FD_checkpoint/convnext_tiny/fold=2-best.pth',
                                            'FD_checkpoint/convnext_tiny/fold=3-best.pth',
                                            'FD_checkpoint/convnext_tiny/fold=4-best.pth']}


def predict(model, loader, config):
    model.eval()
    preds = []
    tepoch = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tepoch):
            data = data.to(config.device)
            outputs = model(data)
            preds.append(outputs)
    return torch.cat(preds).sigmoid().cpu().numpy()


def main():
    config = CFG_CSC
    if os.path.exists('/kaggle/input'):
        config.input_dir = '../input/rsna-2022-cervical-spine-fracture-detection'
    # Load data
    test_df = pd.read_csv(f'{config.input_dir}/{config.label_file}')

    if config.debug:
        test_df = test_df.iloc[:100]

    test_trainsform = build_transform(config.image_size, is_train=False, include_top=True)
    test_dataset = RSNAClassificationDataset(image_dir=f"{config.input_dir}/{config.image_folder}", df=test_df,
                                             img_cols=config.img_cols, label_cols=config.label_cols,
                                             img_format='png', transform=test_trainsform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             num_workers=config.num_workers, pin_memory=config.pin_memory,
                             shuffle=False)

    # Load model
    models = []
    for model_name, checkpoint_dirs in config.CSC_checkpoint_dirs.items():
        for checkpoint_dir in checkpoint_dirs:
            # Initialize model
            model = RSNAClassifier(model_name, pretrained=config.pretrained,
                                   checkpoint_path=config.checkpoint_path, 
                                   in_chans=config.in_chans, num_classes=config.num_classes,
                                   drop_path_rate=config.drop_path_rate)
            model = model.to(config.device)

            # Load weights
            checkpoint = load_checkpoint(checkpoint_dir)
            if 'auc' in checkpoint:
                print(f"AUC: {checkpoint['auc']}")
            model.load_state_dict(checkpoint['model'])
            models.append(model)

    # Predict
    preds = []
    for model in models:
        preds.append(predict(model, test_loader, config))
    preds = np.mean(preds, axis=0)
    # preds = (preds > 0.5).astype('int')
    test_df = test_dataset.df
    test_df[config.label_cols] = preds
    # test_df.sort_values(config.img_cols, inplace=True)
    test_df.to_csv(config.out_file, index=False, mode=config.mode, 
                   header=True if config.mode=='w' else not os.path.exists(config.out_file))
    print(test_df)


if __name__ == '__main__':
    main()