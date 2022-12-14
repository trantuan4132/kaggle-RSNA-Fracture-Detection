import os, glob, zipfile
import cv2
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import re
import json
import pydicom as dicom
import nibabel as nib


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_dicom(path):
    """Load image from dicom file"""
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def load_nii(path):
    """Load image from nii file"""
    ex = nib.load(path)
    ex = ex.get_fdata()  # convert to numpy array
    ex = ex[:, ::-1, ::-1].transpose(2, 1, 0)  # align orientation with train image
    # ex = np.where(ex>0, 255, 0)
    return ex.astype(np.uint8)


def square_bbox(x0, y0, x1, y1, low=0, high=1):
    w, h = x1 - x0, y1 - y0
    w_pad, h_pad = h - min(h, w), w - min(h, w)
    return np.array([x0-w_pad//2, y0-h_pad//2, x1+w_pad//2, y1+h_pad//2]).clip(low, high)


def load_image(image_file, to_RGB=True, crop_coord=[]):
    """Load and convert image to RGB"""
    if os.path.exists(os.path.dirname(image_file)):
        if image_file[image_file.rfind('.')+1:] == 'dcm':
            image = load_dicom(image_file)
        else:
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    elif os.path.exists(f'{os.path.dirname(image_file)}.zip'):
        subdir = os.path.dirname(image_file)
        with zipfile.ZipFile(f'{subdir}.zip', "r") as z:
            buf = z.read(f"{os.path.basename(subdir)}/{os.path.basename(image_file)}")
            np_buf = np.frombuffer(buf, np.uint8)
            image = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
    else:
        raise Exception('No images found')
    if to_RGB and (len(image.shape) < 3 or image.shape[2] < 3):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(crop_coord) >= 4:
        # x0, y0, x1, y1 = (crop_coord[:4]/320*image.shape[0]).astype('int')
        x0, y0, x1, y1 = (crop_coord[:4]*image.shape[0]).astype('int')
        x0, y0, x1, y1 = square_bbox(x0, y0, x1, y1, low=0, high=image.shape[0])
        image = image[y0: y1, x0: x1]
    return image


class Struct:
    """Convert dict to object"""
    def __init__(self, **entries): 
        self.__dict__.update(entries)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_checkpoint(checkpoint_path=None, fold=None, checkpoint_dir=None, postfix=''):
    """
    Loads the checkpoint from the checkpoint_path or the latest checkpoint from the checkpoint_dir
    
    Args:
    -----
    checkpoint_path: str
        Path to the checkpoint
    fold: int
        Fold number
    checkpoint_dir: str
        Path to the checkpoint directory
    postfix: str
        Postfix to add to the checkpoint name
    """
    checkpoint = None
    if checkpoint_path:
        # Load checkpoint given by the path
        if checkpoint_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, 
                                                            map_location='cpu', 
                                                            check_hash=True)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from {checkpoint_path}")
    elif checkpoint_dir and fold is not None:
        # Load checkpoint from the latest one
        checkpoint_files = glob.glob(f"{checkpoint_dir}/fold=*-epoch=*{postfix}.pth")
        checkpoint_files = {f: int(re.search('epoch=(\d+)', f).group(1)) for f in checkpoint_files 
                            if int(re.search('fold=(\d+)', f).group(1)) == fold}
        if len(checkpoint_files) > 0:
            checkpoint_file = max(checkpoint_files, key=checkpoint_files.get)
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
    return checkpoint

def save_checkpoint(checkpoint, save_path):
    """Saves the checkpoint to the save_path"""
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    torch.save(checkpoint, save_path)

def log_to_file(log_stats, log_file, checkpoint_dir):
    """Saves the log to the log_file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(f"{checkpoint_dir}/{log_file}", mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")