import pandas as pd
import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_predict_dir', type=str, default='./yolov5/runs/exp/labels/',
                        help='Path to the yolo txt output')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Path to save the results')

    return parser.parse_args()

def to_csv(predict_dir, save_dir):
    txt_files_list = os.listdir(predict_dir)
    res = pd.DataFrame()
    for i, txt_file in enumerate(txt_files_list):
        # set the path
        path = f'{predict_dir}/{txt_file}'
        df = pd.read_csv(path, sep=" ", header=None)
        id = [txt_file.split('_')[0]]
        slc = [txt_file.split('_')[1].split('.')[0]]
        fracture = [df.iloc[0,0]]
        tmp = pd.DataFrame(data={'StudyInstanceUID': id, 'Slice': slc, 'fracture': fracture})
        res = pd.concat([res, tmp], axis=0, ignore_index=True)
    
    res.to_csv(save_dir)
    return res

if __name__ == '__main__':
    args = parse_args()
    to_csv(args.yolo_predict_dir, args.save_dir)