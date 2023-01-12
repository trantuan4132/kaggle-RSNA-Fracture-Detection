import os, glob
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from utils import load_image


class RSNAClassificationDataset(Dataset):
    def __init__(self, image_dir, df, img_cols, label_cols, img_format='jpg', bbox_label=True, 
                 transform=None, use_2dot5D=True, overlap=True, seq_len=None, crop_cols=[]):
        """
        Args:
        -----
        image_dir: str
            Path to the image directory
        df: pandas dataframe
            Dataframe containing the image ids and labels
        img_cols: list
            List of the names of the column containing the image id
        label_cols: list
            List of the names of the columns containing the labels
        img_format: str, default 'jpg'
            The image file format or extension
        transform: albumentations transform, optional
            Albumentations transform to apply to the image
        """
        super(RSNAClassificationDataset, self).__init__()
        self.image_dir = image_dir
        self.df = df
        self.img_cols = img_cols
        self.label_cols = label_cols
        self.img_format = img_format
        self.bbox_label = bbox_label
        self.transform = transform
        self.overlap = overlap
        self.use_2dot5D = use_2dot5D
        self.seq_len = max(seq_len, 3) if self.use_2dot5D and seq_len else seq_len
        self.crop_cols = crop_cols

        if self.df is None:
            slice_lst = [(uid, sorted([int(f[:f.rfind('.')]) for f in os.listdir(f'{self.image_dir}/{uid}')])) 
                          for uid in os.listdir(self.image_dir)]
            self.df = pd.DataFrame(slice_lst, columns=self.img_cols).explode(self.img_cols[1])

        if self.df is not None and self.img_cols[1] not in self.df.columns:
            slice_lst = [(uid, sorted([int(f[:f.rfind('.')]) for f in os.listdir(f'{self.image_dir}/{uid}')])) 
                          for uid in self.df[self.img_cols[0]]]
            slice_df = pd.DataFrame(slice_lst, columns=self.img_cols).explode(self.img_cols[1])
            self.df = slice_df.merge(self.df, on=self.img_cols[0])

        self.seq_label = True
        if isinstance(self.df[self.img_cols[1]].iloc[0], (list, tuple, np.ndarray)):
            self.df[self.img_cols[1]] = self.df[self.img_cols[1]].apply(lambda x: x + x[-1:]*(seq_len-len(x)))
            self.df = self.df.explode(self.img_cols[1], ignore_index=True)#.reset_index()
            self.seq_label = False
            
        # 2.5D
        def is_start_pos(x, overlap=True, seq_len=1):
            stride = 1 if seq_len <= 3 else max(1, seq_len//8)
            start_ids = np.array([True] * (len(x)-seq_len+1) + [False]*(seq_len-1))
            start_ids[:len(x)-seq_len] = np.arange(0, len(x)-seq_len) % stride == 0
            if not overlap:
                start_ids[:len(x)-seq_len] = np.arange(0, len(x)-seq_len) % seq_len == 0 #np.array([True,False,False] * len(x))[:len(x)-2] + [False]*2)
            return start_ids

        self.start_ids = self.df.groupby(img_cols[0:3:2])[img_cols[1]].transform(is_start_pos, overlap=self.overlap, 
                                                                                 seq_len=self.seq_len or (3 if self.use_2dot5D else 1))
        self.start_ids = self.df[self.start_ids].index.tolist()

    def __len__(self):
        return len(self.start_ids)

    def __getitem__(self, index):
        image_lst, label_lst, bbox_lst, cls_label_lst = [], [], [], []
        for i in range(0, self.seq_len or 1, 3 if self.use_2dot5D else 1):
            if self.use_2dot5D:
                if self.df is not None:
                    rows = self.df.loc[self.start_ids[index] + i:][:3]
                    image_files = [f'{self.image_dir}/{row[self.img_cols[0]]}/{row[self.img_cols[1]]}.{self.img_format}' for _, row in rows.iterrows()]
                    labels = rows[self.label_cols].values.astype('float') \
                            if set(self.label_cols).issubset(self.df.columns) \
                            else np.zeros((3, len(self.label_cols)))
                # else:
                #     image_file = glob.glob(f'{self.image_dir}/*/*.{self.img_format}')[index]
                #     label = np.zeros(len(self.label_cols))

                # Load and concat images
                crop_coords = rows[self.crop_cols].values if self.crop_cols else [[]]*3
                image = cv2.merge([load_image(image_file, to_RGB=False, crop_coord=coord) for image_file, coord in zip(image_files, crop_coords)])
                label = labels.max(axis=0) if self.bbox_label else labels.copy()
                labels = labels[(labels[:, 4:] > 0.5).any(axis=1)]
                bbox, class_labels = ([[*labels[:, :2].min(axis=0), *labels[:, 2:4].max(axis=0)]], [self.label_cols[4]]) \
                                    if self.bbox_label and len(labels) > 0 else ([], [])
            else:
                if self.df is not None:
                    row = self.df.loc[self.start_ids[index] + i]
                    image_file = f'{self.image_dir}/{row[self.img_cols[0]]}/{row[self.img_cols[1]]}.{self.img_format}'
                    label = row[self.label_cols].values.astype('float') \
                            if set(self.label_cols).issubset(self.df.columns) \
                            else np.zeros(len(self.label_cols))
                else:
                    image_file = glob.glob(f'{self.image_dir}/*/*.{self.img_format}')[index]
                    label = np.zeros(len(self.label_cols))

                # Load and convert image to RGB
                crop_coord = row[self.crop_cols].values if self.crop_cols else []
                image = load_image(image_file, to_RGB=False, crop_coord=crop_coord)
                bbox, class_labels = ([label[:4]], [self.label_cols[4]]) if self.bbox_label and (label[4:] > 0.5).any() else ([], [])

            image_lst.append(image), label_lst.append(label)
            bbox_lst.append(bbox), cls_label_lst.append(class_labels)

        image, label = np.stack(image_lst, axis=0), np.concatenate(label_lst, axis=0)
        bbox, class_labels = np.concatenate(bbox_lst, axis=0), np.concatenate(cls_label_lst, axis=0)

        # Apply augmentation
        if self.transform:
            if self.bbox_label:
                transformed = self.transform(image=image[0], bboxes=bbox, class_labels=class_labels,
                                             **dict((f'image{i}', img) for i,img in enumerate(image[1:])))
                if len(transformed['bboxes']) > 0:
                    label[:4] = transformed['bboxes'][0]
            else:
                transformed = self.transform(image=image[0], **dict((f'image{i}', img) for i,img in enumerate(image[1:])))
            if self.seq_len:
                image_lst = [transformed['image']] + [transformed[f'image{i}'] for i in range(len(image[1:]))]
                image = torch.stack(image_lst, 0) if isinstance(image[0], torch.Tensor) else np.stack(image_lst, 0)
            else:
                image = transformed['image']
        if image.shape[0] == 1:
            image = torch.squeeze(image, 0) if isinstance(image[0], torch.Tensor) else np.squeeze(image, 0)
        return (image, label if self.seq_label else label.max(axis=tuple(range(len(label.shape)-1))))


class RandomCircularCrop(ImageOnlyTransform):
    def apply(self, image, **params):
        mask = np.zeros(image.shape, dtype=np.uint8)
        return image * cv2.circle(mask, center=(mask.shape[0]//2, mask.shape[1]//2), 
                                  radius=image.shape[0]//2, color=(1,1,1), thickness=-1)


def build_transform(image_size=None, is_train=True, include_top=True, circular_crop=False, **kwargs):
    """
    Builds a transformations pipeline for the data.

    Args:
    -----
    image_size: int, optional
        The size of the image to be transformed.
    is_train: bool, optional
        Whether the data is being used for training.
    include_top: bool, optional
        Whether to normalize and convert to tensor.
    include_top: bool, optional
        Whether to crop the image in a circle (the pixels outside the circle is blacked out)
    additional_targets: dict, optional
        A dictionary of additional targets to be applied same transformation as the image.
    """
    transform = []
    if image_size:
        transform.append(A.Resize(image_size, image_size))
    image_size = image_size or 40
    if is_train:
        transform.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.5),
            # A.OneOf([
            #   A.ImageCompression(),
            #   A.Downscale(scale_min=0.1, scale_max=0.15),
            # ], p=0.2),
            # A.PiecewiseAffine(p=0.2),
            # A.Sharpen(p=0.2),
            # RandomCircularCrop(p=1.0),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.5),
            # A.CoarseDropout(max_height=int(image_size*0.2), max_width=int(image_size*0.2), 
            #                 min_holes=1, max_holes=4, p=0.5),
            A.Cutout(num_holes=4, max_h_size=int(image_size*0.2), 
                     max_w_size=int(image_size*0.2), fill_value=0, p=0.5),
        ])
    if circular_crop:
        transform.append(RandomCircularCrop(p=1.0))
    if include_top:
        transform.extend([
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])
    return A.Compose(transform, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Configuration
    stage = 1
    input_dir = 'dataset'
    image_dir = f'{input_dir}/train_images'
    use_2dot5D = True
    n_rows = 3
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

    if stage == 1:
        img_cols = ['StudyInstanceUID', 'Slice']
        crop_cols = None
        label_cols = ['x0', 'y0', 'x1', 'y1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        df = pd.read_csv(f'{input_dir}/train_vert_bbox_fold5.csv')
        bbox_label = True 
        seq_len = None
    elif stage == 2:
        img_cols = ['StudyInstanceUID', 'Slice', 'vertebrae']
        crop_cols = ['x0', 'y0', 'x1', 'y1']
        label_cols = ['fractured']
        df = pd.read_pickle(f'{input_dir}/vertebrae_df_fold5.pkl')
        bbox_label = False
        seq_len = 24

    # Dataset initialization
    kwargs = dict(additional_targets=dict((f'image{i}', 'image') for i in range(seq_len-1 if seq_len else 0)))
    if bbox_label: 
        kwargs['bbox_params'] = A.BboxParams(format='albumentations', label_fields=['class_labels'])
    transform = build_transform(image_size=512, is_train=True, include_top=False, **kwargs)
    n_samples = n_rows * n_cols
    sample_dataset = RSNAClassificationDataset(image_dir=image_dir, df=df, 
                                               img_cols=img_cols, label_cols=label_cols, 
                                               img_format='dcm', bbox_label=bbox_label, 
                                               transform=transform, use_2dot5D=use_2dot5D, 
                                               overlap=True, seq_len=seq_len, crop_cols=crop_cols)

    # Visualization
    if stage == 1:
        rnd_ids = np.random.randint(len(sample_dataset), size=n_samples)
        for i in rnd_ids:
            print(sample_dataset.df.loc[sample_dataset.start_ids[i]:][:3])
        for i in range(n_samples):
            img, label = sample_dataset[rnd_ids[i]]
            if (label[4:] > 0.0).any():
                x_min, y_min, x_max, y_max = map(lambda x: int(x*img.shape[0]), label[:4])
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), 3)
            axes.ravel()[i].imshow(img)
            axes.ravel()[i].set_title(str(label))
    elif stage == 2:
        rnd_ids = np.random.randint(len(sample_dataset), size=n_samples)
        # for i in rnd_ids:
        print(sample_dataset.df.loc[sample_dataset.start_ids[1]:][:seq_len])
        img, label = sample_dataset[0]
        for i in range(min(n_samples, len(img))):
            axes.ravel()[i].imshow(img[i], cmap='bone')
            axes.ravel()[i].set_title(str(label[0]))

    # fig.savefig('sample.png')
    plt.show()