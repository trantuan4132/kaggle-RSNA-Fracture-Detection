import os, glob
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from utils import load_dicom


class RSNAClassificationDataset(Dataset):
    def __init__(self, image_dir, df, img_cols, label_cols, img_format='jpg', transform=None):
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
        self.transform = transform

        if self.df is not None and self.img_cols[1] not in self.df.columns:
            slice_lst = [(uid, sorted([int(f[:f.rfind('.')]) for f in os.listdir(f'{self.image_dir}/{uid}')])) 
                          for uid in self.df[self.img_cols[0]]]
            slice_df = pd.DataFrame(slice_lst, columns=self.img_cols).explode(self.img_cols[1])
            self.df = slice_df.merge(self.df, on=self.img_cols[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.df is not None:
            row = self.df.iloc[index]
            image_file = f'{self.image_dir}/{row[self.img_cols[0]]}/{row[self.img_cols[1]]}.{self.img_format}'
            label = row[self.label_cols].values.astype('float') \
                    if set(self.label_cols).issubset(self.df.columns) \
                    else np.zeros(len(self.label_cols))
        else:
            image_file = glob.glob(f'{self.image_dir}/*/*.{self.img_format}')[index]
            label = np.zeros(len(self.label_cols))

        # Load and convert image to RGB
        if self.img_format == 'dcm':
            image = load_dicom(image_file)
        else:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply augmentation
        if self.transform:
            image = self.transform(image=image)['image']
        return (image, label)


def build_transform(image_size=None, is_train=True, include_top=True, additional_targets=None):
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
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.5),
            A.CoarseDropout(max_height=int(image_size*0.2), max_width=int(image_size*0.2), 
                            min_holes=1, max_holes=4, p=0.5),
        ])
    if include_top:
        transform.extend([
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])
    return A.Compose(transform, additional_targets=additional_targets)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_cols = ['StudyInstanceUID', 'Slice']
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    df = pd.read_csv('train_CSC.csv')#.iloc[[1994, 2601, 8783]]
    n_rows = 2
    n_cols = 5
    transform = build_transform(image_size=512, is_train=True, include_top=False)
    n_samples = n_rows * n_cols
    sample_dataset = RSNAClassificationDataset(image_dir="train_images", df=df.sample(n_samples), 
                                               img_cols=img_cols, label_cols=label_cols, 
                                               img_format='png', transform=transform)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i in range(n_samples):
        img, label = sample_dataset[i]
        axes.ravel()[i].imshow(img)
    # fig.savefig('sample.png')
    plt.show()