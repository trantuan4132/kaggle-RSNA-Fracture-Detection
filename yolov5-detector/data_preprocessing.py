import os
import pandas as pd
import numpy as np
import argparse
import shutil
import yaml
import tqdm.notebook as tq
from sklearn.model_selection import KFold, StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='train_images',
                        help='Path to the train image directory')
    parser.add_argument('--test_image_dir', type=str, default='test_images',
                        help='Path to the test image directory')
    parser.add_argument('--metadata', type=str, default='metadata.csv',
                        help='Path to the image metadata file')
    parser.add_argument('--bounding_boxes', type=str, default='train_bounding_boxes.csv',
                        help='Path to the bounding boxes infomation file')
    parser.add_argument('--yolo_dir', type=str, default='yolov5',
                        help='Path to the yolo directory')

    return parser.parse_args()


def create_study_slice(row):

    study_id = str(row['StudyInstanceUID'])
    slice_num = str(row['slice_number'])

    study_slice = study_id + '_' + slice_num

    return study_slice


def create_yaml_yolo():
    yaml_dict = {
        'train': 'base_dir/images/train',       # path to the train folder
        'val': 'base_dir/images/validation',    # path to the val folder
        'nc': 2,                                # number of classes
        'names': ['0', '1']                     # list of label names
    }
    with open(r'yolov5/my_data.yaml', 'w') as file:
        documents = yaml.dump(yaml_dict, file)


def yolo_setup(yolo_dir):
    '''
        Create a new directory (this is happening inside the yolov5 directory)
        # base_dir
            # images
                # train (contains image files)
                # validation (contains image files)
            # labels
                # train (contains .txt files)
                # validation (contains .txt files)
    '''
    base_dir = os.path.join(yolo_dir, 'base_dir')
    if os.path.isdir(base_dir):
      shutil.rmtree(base_dir)
      os.mkdir(base_dir)
    else:
      os.mkdir(base_dir)

    # images
    images = os.path.join(base_dir, 'images')
    os.mkdir(images)

    # labels
    labels = os.path.join(base_dir, 'labels')
    os.mkdir(labels)

    # create new folders inside images
    train = os.path.join(images, 'train')
    os.mkdir(train)
    validation = os.path.join(images, 'validation')
    os.mkdir(validation)

    # create new folders inside labels
    train = os.path.join(labels, 'train')
    os.mkdir(train)
    validation = os.path.join(labels, 'validation')
    os.mkdir(validation)
    create_yaml_yolo()

def split_fold(num_fold, df):
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=101)

    for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.target)):
        df.loc[val_ , "fold"] = fold
    
    df_train = df[df['fold'] != 0]
    df_val = df[df['fold'] == 0]

    return df_train, df_val
    

def process_data_for_yolo(df, images_dir, data_type='train'):

    for _, row in tq.tqdm(df.iterrows(), total=len(df)):

        # Get the target
        target = row['target']

        # Create the image file name
        study_slice = row['study_slice']
        fname = study_slice + '.png'

        # Only create txt files for class 1 images
        if target == 1:

            # Get the list of bboxes on the image.
            # Each item in the list is a dict containing the image coords.
            bbox_dict = row['boxes']

            # put the coords into a list
            bbox_list = [bbox_dict]

            # These are the original image sizes.
            # If we have resized the images then this must be changed to
            # the new sizes. We will then also be using resized bbox coords.
            image_width = row['w']
            image_height = row['h']

            # Convert into the Yolo input format
            # ...................................
            yolo_data = []

            # row by row
            for coord_dict in bbox_list:

                xmin = int(coord_dict['x'])
                ymin = int(coord_dict['y'])
                bbox_w = int(coord_dict['width'])
                bbox_h = int(coord_dict['height'])

                # We only have one class i.e. opacity
                # We will set the class_id to 0 for all images.
                # Class numbers must start from 0.
                class_id = target

                x_center = xmin + (bbox_w/2)
                y_center = ymin + (bbox_h/2)

                # Normalize
                # Yolo expects the dimensions to be normalized i.e.
                # all values between 0 and 1.

                x_center = x_center/image_width
                y_center = y_center/image_height
                bbox_w = bbox_w/image_width
                bbox_h = bbox_h/image_height

                # [class_id, x-center, y-center, width, height]
                yolo_list = [class_id, x_center, y_center, bbox_w, bbox_h]

                yolo_data.append(yolo_list)

            # convert to nump array
            yolo_data = np.array(yolo_data)

            # Write the image bbox info to a txt file
            #image_id = image_name.split('.')[0]
            np.savetxt(os.path.join('yolov5/base_dir',
                                    f"labels/{data_type}/{study_slice}.txt"),
                        yolo_data,
                        fmt=["%d", "%f", "%f", "%f", "%f"]
                       )  # fmt means format the columns

        # Copy the image to images
        # Set the path to the images here.
        shutil.copyfile(
            f"{images_dir}/{fname}",
            os.path.join('yolov5/base_dir', f"images/{data_type}/{fname}")
        )


def process_metadata(metadata, bounding_boxes):
    metadata = metadata.copy(deep=True)
    metadata['target'] = metadata(metadata['label'])
    bounding_boxes = bounding_boxes.copy(deep=True)
    bounding_boxes['study_slice'] = bounding_boxes.apply(create_study_slice, axis=1)
    bounding_boxes = bounding_boxes.set_index('study_slice')
    bbox_list = []

    for i in range(0, len(metadata)):

        target = metadata.loc[i, 'target']
        study_slice = metadata.loc[i, 'study_slice']

        if target == 1:

            x = bounding_boxes.loc[study_slice, 'x']
            y = bounding_boxes.loc[study_slice, 'y']
            width = bounding_boxes.loc[study_slice, 'width']
            height = bounding_boxes.loc[study_slice, 'height']

            bbox_dict = {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }

            bbox_list.append(bbox_dict)

        else:
            bbox_list.append('none')

    # Add the bbox_list to metadata
    metadata['boxes'] = bbox_list
    return metadata, bounding_boxes

if __name__ == "__main__":
    args = parse_args()

    # Create a directory structure inside the yolov5 folder
    yolo_setup(args.yolo_dir)

    # Load meta data
    metadata = pd.read_csv(args.metadata)
    bounding_boxes = pd.read_csv(args.bounding_boxes)
    metadata, bounding_boxes = process_metadata(metadata, bounding_boxes)
    df_train, df_val = split_fold(5, metadata)

   # Preprocess data
    if args.train:
        process_data_for_yolo(df_train, args.train_image_dir)
        process_data_for_yolo(df_val, args.train_image_dir, data_type='validation')
    else:
        process_data_for_yolo(metadata, args.train_image_dir, False)
