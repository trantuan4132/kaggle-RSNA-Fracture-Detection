import os, glob
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import pydicom as dicom
import nibabel as nib
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='train_images',
                        help='Path to the image directory')
    parser.add_argument('--label_path', type=str, default='train.csv',
                        help='Path to the label file')
    parser.add_argument('--relabel_path', type=str, default='relabel.csv',
                        help='Path to the relabel file')
    parser.add_argument('--vert_label_path', type=str, default='train_CSC_full.csv',
                        help='Path to the vertebrae label file')
    parser.add_argument('--seg_dir', type=str, default='segmentations',
                        help='Path to the segmentation directory')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Path to the output directory')                    
    parser.add_argument('--drop', action='store_true',
                        help='Drop the image id contained in the relabel file')
    parser.add_argument('--get_rvs_lst', action='store_true',
                        help='Extract patient id with their scan in reverse order')
    parser.add_argument('--get_vert_label', action='store_true',
                        help='Extract vertebrae labels from segmentation')
    parser.add_argument('--get_frac_label', action='store_true',
                        help='Create fracture labels')
    return parser.parse_args()


def is_reverse(path):
    """Check if the scan is in reverse direction (feet → head instead of head → feet)"""
    paths = glob.glob(f'{path}/*')
    paths.sort(key=lambda x:int(os.path.basename(x)[:-len('.dcm')]))
    z_first = dicom.dcmread(paths[0]).get("ImagePositionPatient")[-1]
    z_last = dicom.dcmread(paths[-1]).get("ImagePositionPatient")[-1]
    if z_last < z_first:
        return False
    return True


def get_reverse_list(directory='train_images'):
    """Extract patient id with their scan in reverse order"""
    reverse_lst = []
    for path in tqdm(glob.glob(f"{directory}/*")):
        if is_reverse(path):
            reverse_lst.append(os.path.basename(path))
    reverse_lst.sort()
    return reverse_lst


def drop_data(df, df_relabel, img_col):
    """Drop data to be ignored during training"""
    df = df.set_index(img_col).drop(df_relabel[img_col].unique()).reset_index()
    return df


def load_nii(path):
    """Load image from nii file"""
    ex = nib.load(path)
    ex = ex.get_fdata()  # convert to numpy array
    ex = ex[:, ::-1, ::-1].transpose(2, 1, 0)  # align orientation with train image
    # ex = np.where(ex>0, 255, 0)
    return ex.astype(np.uint8)


def extract_vertebrae_labels_from_segmentation(segmentation_dir='segmentations', image_dir='train_images', reverse_lst=[]):
    """Extract vertebrae labels from segmentation"""
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    def extract(path, image_dir='train_images', reverse_lst=[]):
        def remove_outlier(seg_labels, direct='td'):
            cnt = {}
            start, step, end = (1, 1, 7) if direct=='td' else (7, -1, 1)
            for idx in (range(len(seg)) if direct=='td' else range(len(seg)-1, -1, -1)):
                new_seg_label = []
                cd1 = start in seg_labels[idx]
                cd2 = start+step in seg_labels[idx] and cnt.get(start, 0) > 0
                if cd1 or cd2:
                    if cd2 and not cd1:
                        start += step
                    s = start
                    while True:
                        cnt[s] = cnt.get(s, 0) + 1
                        new_seg_label.append(s)
                        s += step
                        if s not in seg_labels[idx]:
                            break
                elif start == end:
                    start += step
                seg_labels[idx] = new_seg_label
            return seg_labels
        
        # Load segmentation data
        seg = load_nii(path)

        # Get list of slices from directory
        uid = os.path.basename(path)[:-len('.nii')]
        slice_lst = sorted([int(f[:f.rfind('.')]) for f in os.listdir(f"{image_dir}/{uid}")], 
                           reverse=uid in reverse_lst)
        
        # Get list of labels each slice
        seg_labels = []
        for idx in range(len(seg)):
            seg_labels.append(set(seg[idx].flatten()) & set(range(1, 8)))
            
        # Remove outliers
        seg_labels = remove_outlier(seg_labels, direct='td')    # Top-down
        seg_labels = remove_outlier(seg_labels, direct='bu')    # Bottom up
            
        # Assign corresponding labels for each slice
        seg_lst = []
        for idx, slice in enumerate(slice_lst):
            labels = np.zeros(7, dtype=int)
            if len(seg_labels[idx]) > 0:
                labels[np.array(list(seg_labels[idx]))-1] = 1
            seg_lst.append([uid, slice] + list(labels))
        return seg_lst
                
    seg_lst = Parallel(n_jobs=-1)(
        delayed(extract)(path, image_dir, reverse_lst)
        for path in tqdm(glob.glob(f"{segmentation_dir}/*")) if path.endswith(".nii")
    )
    seg_lst = sum(seg_lst, [])

    seg_df = pd.DataFrame(seg_lst, columns=["StudyInstanceUID", "Slice"] + label_cols)
    seg_df = seg_df.sort_values(by=['StudyInstanceUID', 'Slice'])
    return seg_df


def create_fracture_labels(df, df_vert, img_col, label_cols):
    """Create fracture labels from train labels and vertebrae labels"""
    vert_label_cols = [col + '_vert' for col in label_cols]
    df_frac = df_vert.set_index(img_col).join(df.set_index(img_col), lsuffix='_vert').reset_index()
    df_frac.loc[:, label_cols] = df_frac.loc[:, label_cols] * df_frac.loc[:, vert_label_cols]
    return df_frac.drop(columns=vert_label_cols)


if __name__ == "__main__":
    args = parse_args()
    img_col = 'StudyInstanceUID'
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    reverse_lst = None

    # Get list of scan in reverse order
    if args.get_rvs_lst:
        reverse_lst = get_reverse_list(args.image_dir)
        with open(f"{args.out_dir}/reverse_list.txt", 'w') as f:
            f.write("\n".join(sorted(reverse_lst)))
    
    # Drop data to be ignored during training
    if args.drop:
        df = pd.read_csv(args.label_path)
        print('Number of rows before dropping:', len(df))
        df_relabel = pd.read_csv(args.relabel_path)
        df = drop_data(df, df_relabel, img_col)
        print('Number of rows after dropping:', len(df))
        df.to_csv(f"{args.out_dir}/{os.path.basename(args.label_path)}", index=False)

    # Extract vertebrae labels from segmentation
    if args.get_vert_label:

        # Get list of scan in reverse order if args.get_rvs_lst is False
        if reverse_lst is None:
            if os.path.exists(f"{args.out_dir}/reverse_list.txt"):
                with open(f"{args.out_dir}/reverse_list.txt", mode='r') as f:
                    reverse_lst = f.read().split("\n")
            else:
                reverse_lst = get_reverse_list(args.image_dir)

        # Extract vertebrae labels
        seg_df = extract_vertebrae_labels_from_segmentation(segmentation_dir=args.seg_dir, 
                                                            image_dir=args.image_dir, 
                                                            reverse_lst=reverse_lst)
        seg_df.to_csv(f"{args.out_dir}/train_CSC.csv", index=False)

    # Create fracture labels
    if args.get_frac_label:
        df = pd.read_csv(args.label_path)
        df_vert = pd.read_csv(args.vert_label_path)
        df_frac = create_fracture_labels(df, df_vert, img_col, label_cols)
        df_frac.to_csv(f"{args.out_dir}/train_FD.csv", index=False)