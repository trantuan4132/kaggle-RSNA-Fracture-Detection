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
    parser.add_argument('--out_dir', type=str, default='output',
                        help='Path to the output directory')
    parser.add_argument('--out_file', type=str, default=None,
                        help='Name of the output file, default output file is used if it is not specified')                      
    parser.add_argument('--drop', action='store_true',
                        help='Drop the image id contained in the relabel file')
    parser.add_argument('--get_rvs_lst', action='store_true',
                        help='Extract patient id with their scan in reverse order')
    parser.add_argument('--get_vert_label', action='store_true',
                        help='Extract vertebrae labels from segmentation')
    parser.add_argument('--get_vert_bbox', action='store_true',
                        help='Extract vertebrae bounding box annotation from segmentation')
    parser.add_argument('--get_frac_label', action='store_true',
                        help='Create fracture labels')
    parser.add_argument('--seq_len', type=int, default=None, 
                        help='Length of the sequence of images to be sampled for each vertebrae')
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


def drop_data(df, relabel_df, img_col):
    """Drop data to be ignored during training"""
    df = df.set_index(img_col).drop(relabel_df[img_col].unique()).reset_index()
    return df


def load_nii(path):
    """Load image from nii file"""
    ex = nib.load(path)
    ex = ex.get_fdata()  # convert to numpy array
    ex = ex[:, ::-1, ::-1].transpose(2, 1, 0)  # align orientation with train image
    # ex = np.where(ex>0, 255, 0)
    return ex.astype(np.uint8)


def extract_vertebrae_labels_from_segmentation(segmentation_dir='segmentations', image_dir='train_images', seg_label2idx=None, vert_ratio=True, reverse_lst=[]):
    """Extract vertebrae labels from segmentation"""
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    if seg_label2idx is None:
        seg_label2idx = {i+1: i for i in range(7)}
    
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
        uid = os.path.basename(path)
        uid = uid[:uid.rfind('.nii')]
        slice_lst = sorted([int(f[:f.rfind('.')]) for f in os.listdir(f"{image_dir}/{uid}")], 
                           reverse=uid in reverse_lst)
        
        # Get list of labels each slice
        seg_labels = []
        for idx in range(len(seg)):
            seg_labels.append(set(seg[idx].flatten()) & set(seg_label2idx.keys()))
            
        # Remove outliers
        seg_labels = remove_outlier(seg_labels, direct='td')    # Top-down
        seg_labels = remove_outlier(seg_labels, direct='bu')    # Bottom up
            
        # Assign corresponding labels for each slice
        seg_lst = []
        for idx, slice in enumerate(slice_lst):
            labels = np.zeros(7, dtype=int)
            if len(seg_labels[idx]) > 0:
                labels[[seg_label2idx[l] for l in seg_labels[idx]]] = [(seg[idx]==l).sum() for l in seg_labels[idx]] if vert_ratio else 1
            seg_lst.append([uid, slice] + list(labels))
        return seg_lst
                
    seg_lst = Parallel(n_jobs=-1)(
        delayed(extract)(path, image_dir, reverse_lst)
        for path in tqdm(glob.glob(f"{segmentation_dir}/*")) if path.endswith((".nii", ".nii.gz"))
    )
    seg_lst = sum(seg_lst, [])

    seg_df = pd.DataFrame(seg_lst, columns=["StudyInstanceUID", "Slice"] + label_cols)
    seg_df = seg_df.sort_values(by=['StudyInstanceUID', 'Slice'])
    if vert_ratio:
        seg_df[label_cols] = seg_df[label_cols] / seg_df.groupby('StudyInstanceUID')[label_cols].transform(max)
    return seg_df


def extract_vertebrae_bbox_annotation(segmentation_dir='segmentations', idx2seg_label=None, seg_df=None, reverse_lst=[]):
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    if idx2seg_label is None:
        idx2seg_label = {i: i+1 for i in range(7)}

    def extract(path, reverse_lst=[]):
        # Load segmentation data
        seg = load_nii(path)

        # Get list of slices
        uid = os.path.basename(path)
        uid = uid[:uid.rfind('.nii')]
        tmp_df = seg_df.query(f"StudyInstanceUID=='{uid}'").sort_values('Slice', ascending=not uid in reverse_lst)
        slice_lst = tmp_df['Slice'].values

        # Get list of labels each slice
        seg_labels = [np.where(label > 0)[0] for label in tmp_df[label_cols].values]

        # Assign corresponding labels for each slice
        bbox_lst = []
        for idx, slice in enumerate(slice_lst):
            if len(seg_labels[idx]) == 0:
                bbox_lst.append([uid, slice, 0, 0, 0, 0, 0])
            else:
                rows, cols = np.where(np.isin(seg[idx], np.vectorize(idx2seg_label.get)(seg_labels[idx])))
                bbox = np.array([cols.min(), rows.min(), cols.max()+1, rows.max()+1])
                bbox = bbox / np.array([seg[idx].shape[1], seg[idx].shape[0]]*2)
                # rows, cols = rows / seg[idx].shape[0], cols / seg[idx].shape[1]
                bbox_lst.append([uid, slice, *bbox, 1])
        return bbox_lst
        
    bbox_lst = Parallel(n_jobs=-1)(
        delayed(extract)(path, reverse_lst)
        for path in tqdm(glob.glob(f"{segmentation_dir}/*")) if path.endswith((".nii", ".nii.gz"))
    )
    bbox_lst = sum(bbox_lst, [])

    bbox_df = pd.DataFrame(bbox_lst, columns=["StudyInstanceUID", "Slice", "x0", "y0", "x1", "y1", "vertebrae"])
    bbox_df = bbox_df.sort_values(by=['StudyInstanceUID', 'Slice'])
    return bbox_df


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


def create_fracture_labels(df, vert_df, img_cols=['StudyInstanceUID', 'Slice'],
                           label_cols=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                           crop_cols=['x0', 'y0', 'x1', 'y1'], extra_cols=[], 
                           vert_col='vertebrae', target_col='fractured', vert_thresh=0.5, seq_len=24):
    """Create fracture labels from train labels and vertebrae labels"""
    if seq_len: 
        # Assign fracture label to each sequence of images

        # Fill in default value if label columns are not part of df
        if not set(label_cols).issubset(df.columns):
            df[label_cols] = 0

        # Get each label columns into one rows
        df = df[img_cols[:1] + label_cols + extra_cols].melt(id_vars=img_cols[:1] + extra_cols, 
                                                             var_name=vert_col, value_name=target_col)

        # Get bounding box coordination and a sequence of slices for each vertebrae in each study
        slice_range_dfs = [vert_df.groupby(img_cols[0]).apply(window_range, label_col=col, thresh=vert_thresh, all_slices=True) for col in label_cols]
        vert_df = vert_df[(vert_df[label_cols] > vert_thresh).any(axis=1)].groupby(img_cols[0])[crop_cols[0:2]].min().join([
            vert_df[(vert_df[label_cols] > vert_thresh).any(axis=1)].groupby(img_cols[0])[crop_cols[2:4]].max(),
            *[pd.DataFrame(np.expand_dims(slice_range_dfs[i], 1).tolist(), columns=[f'{col}'], 
                            index=slice_range_dfs[i].index) for i, col in enumerate(label_cols)],
            vert_df.groupby(img_cols[0])[img_cols[1]].max().rename('max_slice')
        ]).reset_index()                                                 
        # vert_df.rename(columns={f'{col}_slices': col for col in label_cols}, inplace=True)
        vert_df = vert_df[img_cols[:1] + crop_cols + label_cols].melt(id_vars=img_cols[:1] + crop_cols, 
                                                                      var_name=vert_col, value_name=img_cols[1])
        # Select only rows with number of slices >= 5
        vert_df = vert_df[vert_df[img_cols[1]].apply(len) >= 5]

        # Get new list of slices with evenly spaced index from the previous list
        vert_df[img_cols[1]] = vert_df[img_cols[1]].apply(lambda x: np.array(x)[np.linspace(0, len(x)-1, 
                                                                                            seq_len, dtype=int)])
        return df.merge(vert_df, how='inner')
    else:
        # Assign fracture label to each image
        vert_label_cols = [col + '_vert' for col in label_cols]
        frac_df = vert_df.set_index(img_cols[0]).join(df.set_index(img_cols[0]), lsuffix='_vert').reset_index()
        frac_df[vert_label_cols] = (frac_df[vert_label_cols] > 0.5).astype('int')
        frac_df[label_cols] = frac_df[label_cols].values * frac_df[vert_label_cols].values
        return frac_df.loc[:, ~frac_df.columns.str.endswith('_vert')]


if __name__ == "__main__":
    args = parse_args()
    img_cols = ['StudyInstanceUID', 'Slice']
    label_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    reverse_lst = None

    # Get list of scan in reverse order
    if args.get_rvs_lst:
        reverse_lst = get_reverse_list(args.image_dir)
        if not args.out_file: args.out_file = "reverse_list.txt" 
        with open(f"{args.out_dir}/{args.out_file}", 'w') as f:
            f.write("\n".join(sorted(reverse_lst)))
    
    # Drop data to be ignored during training
    if args.drop:
        df = pd.read_csv(args.label_path)
        print('Number of rows before dropping:', len(df))
        relabel_df = pd.read_csv(args.relabel_path)
        df = drop_data(df, relabel_df, img_cols[0])
        print('Number of rows after dropping:', len(df))
        if not args.out_file: args.out_file = os.path.basename(args.label_path)
        df.to_csv(f"{args.out_dir}/{args.out_file}", index=False)

    # Extract vertebrae labels from segmentation
    if args.get_vert_label or args.get_vert_bbox:

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
                                                            seg_label2idx={i+1: i for i in range(7)}, # {41-i: i for i in range(7)}
                                                            reverse_lst=reverse_lst)
        if not args.out_file: args.out_file = "train_CSC.csv" 
        seg_df.to_csv(f"{args.out_dir}/{args.out_file}", index=False)

    # Extract vertebrae bounding box annotation from segmentation
    if args.get_vert_bbox:
        bbox_df = extract_vertebrae_bbox_annotation(segmentation_dir=args.seg_dir,
                                                    idx2seg_label={i: i+1 for i in range(7)},
                                                    seg_df=seg_df)
        if not args.out_file: args.out_file = "train_vert_bbox.csv"                                             
        bbox_df.to_csv(f"{args.out_dir}/{args.out_file}", index=False)

    # Extract both vertebrae labels and bounding box annotation from segmentation to a single file
    if args.get_vert_label and args.get_vert_bbox:
        if not args.out_file: args.out_file = "train_vert_bbox_ratio.csv" 
        seg_df.merge(bbox_df).to_csv(f"{args.out_dir}/{args.out_file}", index=False)

    # Create fracture labels
    if args.get_frac_label:
        df = pd.read_csv(args.label_path)
        vert_df = pd.read_pickle(args.vert_label_path) \
                  if args.vert_label_path.endswith('.pkl') \
                  else pd.read_csv(args.vert_label_path)
        frac_df = create_fracture_labels(df, vert_df, img_cols=img_cols, label_cols=label_cols,
                                         crop_cols=['x0', 'y0', 'x1', 'y1'], extra_cols=[], 
                                         vert_col='vertebrae', target_col='fractured', seq_len=args.seq_len)
        if not args.out_file: 
            if args.seq_len:
                args.out_file = "vertebrae_df.pkl" 
            else:
                args.out_file = "train_CSC_FD.csv" 
                
        frac_df.to_pickle(f"{args.out_dir}/{args.out_file}") \
        if args.out_file.endswith('.pkl') \
        else frac_df.to_csv(f"{args.out_dir}/{args.out_file}", index=False)