import os, argparse
import yaml
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import *
from model import RSNAClassifier
from utils import load_checkpoint, Struct, create_fracture_labels


def predict(model, loader, config):
    model.eval()
    preds = []
    tepoch = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tepoch):
            data = data.to(config.device)
            outputs = model(data)
            preds.append(outputs)
    if config.bbox_label:
        preds = torch.cat(preds)
        if not config.vert_ratio: 
            preds[..., 4:] = preds[..., 4:].sigmoid()
        return preds.cpu().numpy().clip(0,1)
    return torch.cat(preds).sigmoid().cpu().numpy()


def patient_prediction(df, label_cols=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']):
    c1c7 = np.average(df[label_cols].values, axis=0, weights=df[label_cols].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7) # max(c1c7) # 
    return pd.Series(data=np.concatenate([[pred_patient_overall], c1c7]), index=['patient_overall'] + [f'C{i}' for i in range(1, 8)])


def window_range(rows, label_col='vertebrae', window=5, min_periods=3, center=True, thresh=0.1, all_slices=False):
    vert_probs = rows[label_col].rolling(window, min_periods=min_periods, center=center).mean()
    # vert_probs.plot()
    vert_range = np.where(vert_probs > thresh)[0]
    if all_slices:
        return rows['Slice'].iloc[vert_range].tolist()
    if len(vert_range) == 0:
        return rows['Slice'].iloc[[0,-1]]
    start_slice = rows['Slice'].iloc[vert_range[0]]
    end_slice = rows['Slice'].iloc[vert_range[-1]]
    return [start_slice, end_slice]


def load_models(ckpt_dirs, config):
    models = []
    for model_name, checkpoint_dirs in ckpt_dirs.items():
        for checkpoint_dir in checkpoint_dirs:
            # Initialize model
            model = RSNAClassifier(model_name, pretrained=config.pretrained,
                                   checkpoint_path=config.checkpoint_path, 
                                   in_chans=config.in_chans, num_classes=config.num_classes,
                                   drop_path_rate=config.drop_path_rate, use_seq_layer=config.use_seq_layer)

            # Load weights
            checkpoint = load_checkpoint(checkpoint_dir)
            # if 'auc' in checkpoint:
            #     print(f"AUC: {checkpoint['auc']}")
            model.load_state_dict(checkpoint['model'])
            # if config.parallel:
            #     model = nn.DataParallel(model)
            model = model.to(config.device)
            models.append(model)
    return models


def ensemble(models, test_loader, config):
    preds = []
    for model in models:
        preds.append(predict(model, test_loader, config))
    preds = np.mean(preds, axis=0)
    # preds = (preds > 0.5).astype('int')
    return preds


def predict_all(config, test_df=None):
    if os.path.exists('/kaggle/input'):
        config.input_dir = '../input/rsna-2022-cervical-spine-fracture-detection'
    # Load data
    if test_df is None:
        test_df = pd.read_pickle(f'{config.input_dir}/{config.label_file}') \
                  if config.label_file.endswith('.pkl') \
                  else pd.read_csv(f'{config.input_dir}/{config.label_file}')
    test_df.drop(config.label_cols, axis=1, errors='ignore', inplace=True)
    # test_df = test_df.query(f"StudyInstanceUID=='1.2.826.0.1.3680043.6200'")

    if config.debug:
        test_df = test_df.iloc[:100]

    kwargs = dict(additional_targets=dict((f'image{i}', 'image') for i in range(config.seq_len-1 if config.seq_len else 0)))
    if config.bbox_label: 
        kwargs['bbox_params'] = A.BboxParams(format='albumentations', label_fields=['class_labels'])

    test_trainsform = build_transform(config.image_size, is_train=False, include_top=True, **kwargs)
    test_dataset = RSNAClassificationDataset(image_dir=f"{config.input_dir}/{config.image_dir}", df=test_df,
                                             img_cols=config.img_cols, label_cols=config.label_cols,
                                             img_format=config.img_format, bbox_label=config.bbox_label, 
                                             transform=test_trainsform, use_2dot5D=config.use_2dot5D, 
                                             overlap=False, seq_len=config.seq_len, crop_cols=config.crop_cols)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             num_workers=config.num_workers, pin_memory=config.pin_memory,
                             shuffle=False)

    # Load model
    models = load_models(config.checkpoint_dirs, config)

    # Predict
    pred_df = test_dataset.df.copy()
    pred_df.loc[test_dataset.start_ids, config.label_cols] = ensemble(models, test_loader, config)
    # pred_df.sort_values(config.img_cols, inplace=True)
    if config.name == 'CFG_CSC':
        pred_df.to_csv(config.out_file, index=False)
        test_df = pred_df.copy()
    elif config.name == 'CFG_CSC_FD':
        pred_df = pred_df.groupby(config.img_cols[0]) \
                     .apply(lambda df: patient_prediction(df, config.label_cols, config.label_cols)) \
                     .reset_index()
        print(pred_df)
        test_df = test_df.merge(pred_df, how='left', on=config.img_cols[0]).fillna(0)
        test_df['fractured'] = test_df.apply(lambda row: row[row['prediction_type']], axis=1)
        test_df[['row_id', 'fractured']].to_csv(config.out_file, index=False)
    elif config.name == 'CFG_vert_bbox_ratio':
        pred_df.fillna(method='ffill', inplace=True)
        # pred_df.to_csv('test.csv', index=False)
        slice_range_dfs = [pred_df.groupby(config.img_cols[0]).apply(window_range, label_col=label_col, thresh=0.5, all_slices=True) for label_col in config.label_cols[4:]]
        test_df = pred_df[(pred_df[config.label_cols[4:]] > 0.5).any(axis=1)].groupby(config.img_cols[0])[config.label_cols[0:2]].min().join([
            pred_df[(pred_df[config.label_cols[4:]] > 0.5).any(axis=1)].groupby(config.img_cols[0])[config.label_cols[2:4]].max(),
            *[pd.DataFrame(np.expand_dims(slice_range_dfs[i], 1).tolist(), columns=[f'{label_col}_slices'], 
                           index=slice_range_dfs[i].index) for i, label_col in enumerate(config.label_cols[4:])],
            pred_df.groupby(config.img_cols[0])[config.img_cols[1]].max().rename('max_slice')
        ]).reset_index()
        test_df.to_pickle(config.out_file) if config.label_file.endswith('.pkl') else test_df.to_csv(config.out_file, index=False)
    elif config.name == 'CFG_FD':
        pred_df.fillna(method='ffill', inplace=True)
        print(pred_df)
        test_df = pred_df.groupby(config.img_cols[0:3:2])[config.label_cols[0]].max().unstack().reset_index()
        test_df.columns = test_df.columns.tolist()
        test_df.fillna(0, inplace=True)
        # test_df = test_df.groupby(config.img_cols[0]).apply(lambda df: patient_prediction(df)).reset_index()
        test_df['patient_overall'] = test_df.apply(lambda x: 1 - np.prod(1 - x[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']] \
                                                                         .sort_values(ascending=False)[:]), axis=1)
    return test_df


def load_config(fname):
    """Load configuration from file"""
    config = lambda: None
    if os.path.exists(fname):
        with open(fname, mode='r') as stream:
            config = Struct(**yaml.safe_load(stream))
        if os.path.exists('/kaggle/input'):
            config.input_dir = '../input/rsna-2022-cervical-spine-fracture-detection'
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.num_classes = len(config.label_cols)
    config.name = os.path.basename(fname).replace('.yaml', '').replace('_infer', '')
    return config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--CFG', type=str, default='config/CFG_FD_infer.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--image_dir', type=str, default=None, 
                        help="Path to the image directory")
    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.CFG)
    if args.image_dir: config.image_dir = args.image_dir
    
    # Generate prediction
    if config.name == 'CFG_vert_bbox_ratio_FD':
        configs = [
            load_config('config/CFG_vert_bbox_ratio_infer.yaml'),
            load_config('config/CFG_FD_infer.yaml')
        ]
        df = pd.read_csv(f'{configs[0].input_dir}/test.csv')
        # Fix inconsistency between test_images and test.csv
        if df.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
            df = pd.DataFrame({
                "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
                "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
                "prediction_type": ["C1", "C1", "patient_overall"]}
            )
        uid_df = pd.DataFrame({
            configs[0].img_cols[0]: df[configs[0].img_cols[0]].unique().tolist()
        })

        test_df = predict_all(config=configs[0], test_df=uid_df)
        print(test_df)
        test_df = create_fracture_labels(uid_df, test_df,
                                    extra_cols=[], vert_col=configs[-1].img_cols[2],
                                    target_col=configs[-1].label_cols[0], seq_len=configs[-1].seq_len)#.to_pickle(configs[1].label_file)
        test_df = predict_all(config=configs[-1], test_df=test_df)
        # print(test_df)
        test_df = df.merge(test_df, how='left', on=configs[-1].img_cols[0]).fillna(0)
        test_df['fractured'] = test_df.apply(lambda row: row[row['prediction_type']], axis=1)
        test_df[['row_id', 'fractured']].to_csv(configs[-1].out_file, index=False)
    else:
        test_df = predict_all(config=config)
    print(test_df)


if __name__ == '__main__':
    main()