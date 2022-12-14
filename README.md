# kaggle-RSNA-Fracture-Detection

![Methodology](docs/Methodology.png)

## Installation

```
git clone https://github.com/trantuan4132/kaggle-RSNA-Fracture-Detection
cd kaggle-RSNA-Fracture-Detection
```

## Set up environment

```
pip install -r requirements.txt
```

## Data

The dataset is available for downlading in a kaggle competition namely [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection). To continue for the next step, the dataset needs to be downloaded in the `./dataset` directory

## Stage 1: Vertebrae localization and ratio regression

![Stage 1](docs/Stage1.png)

### Data Preparation

In this step, images used for training will be those with segmentation mask available, which come from 87 patients out of 2019 total number of patients. The labels used in this step include top left coordination `(x0, y0)` and bottom right coordination `(x1, y1)` of the bounding box that cover all vertebrae C1-C7 in each image. The top left coordination `(x0, y0)` can be extracted from segmentation mask by determining the first column from top to bottom and the first row from left to right that contain the element belong to vertebrae C1-C7, while the bottom right coordination `(x1, y1)` can be determined in the similar way except for using the last column and row instead of the first one. The labels also include the ratio of each vertebrae in each image by counting the number of elements belong to each vertebrae in the segmentation mask, then divide by the maximum number of elements belong to each vertebrae in all images of each patient.

To prepare the label file, run `utils/preprocess_data.py --image_dir dataset/train_images --label_path dataset/train.csv --seg_dir dataset/segmentations --get_vert_label --get_vert_bbox` and the label file `train_vert_bbox_ratio.csv` will be generated in the `./output` directory

```
usage: process_data.py [-h] [--image_dir IMAGE_DIR] [--label_path LABEL_PATH] [--relabel_path RELABEL_PATH] [--vert_label_path VERT_LABEL_PATH] [--seg_dir SEG_DIR]
                       [--out_dir OUT_DIR] [--drop] [--get_rvs_lst] [--get_vert_label] [--get_vert_bbox] [--get_frac_label] [--seq_len SEQ_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to the image directory
  --label_path LABEL_PATH
                        Path to the label file
  --relabel_path RELABEL_PATH
                        Path to the relabel file
  --vert_label_path VERT_LABEL_PATH
                        Path to the vertebrae label file
  --seg_dir SEG_DIR     Path to the segmentation directory
  --out_dir OUT_DIR     Path to the output directory
  --drop                Drop the image id contained in the relabel file
  --get_rvs_lst         Extract patient id with their scan in reverse order
  --get_vert_label      Extract vertebrae labels from segmentation
  --get_vert_bbox       Extract vertebrae bounding box annotation from segmentation
  --get_frac_label      Create fracture labels
  --seq_len SEQ_LEN     Length of the sequence of images to be sampled for each vertebrae
```


### Fold Splitting

Next, run `utils/fold_split.py --label_path output/train_vert_bbox_ratio.csv --multi_label` to split data into 5 folds and the new label file `train_vert_bbox_ratio_fold5.csv` will be saved in the `./output` directory

```
python utils/fold_split.py

usage: fold_split.py [-h] [--label_path LABEL_PATH] [--kfold KFOLD] [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --label_path LABEL_PATH
                        Path to the label file
  --kfold KFOLD         Number of folds
  --save_path SAVE_PATH
                        Path to the save file
```

### Training

For training, run `train.py` to train model (training customization can be done by modifying the configuration file):

```
python train.py --CFG config/CFG_vert_bbox_ratio_train.yaml
```

By default, this will run all 5 folds sequentially. To run a single fold instead of all 5 folds, specify `--fold` parameter with a number between 0 and 4

**Note:** If pretrained weight is not available for downloading from timm by setting `pretrained=True`, try downloading it directly using link provided in timm repo then set `checkpoint_file: <downloaded-weight-file-path>` in the config file to load the weight. In case timm model fails to load due to the different format that the pretrained weight might have, run `utils/preprocess_checkpoint.py` (this will only work when timm provide the `checkpoint_filter_fn` in their implementation for the model specified).

For instance, to train `convnext_tiny`, go to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py and find the corresponding checkpoint link (Ex. https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth) provided inside the file then download into current directory and run `utils/preprocess_checkpoint.py` so a new checkpoint file with `_altered` at the end of the file name will be created.

```
python utils/preprocess_checkpoint.py

usage: preprocess_checkpoint.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--model MODEL] [--variant VARIANT]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file
  --model MODEL         Model name
  --variant VARIANT     Model variant
```

### Inference

For inference, run `inference.py` to generate prediction (inference customization can be done by modifying the configuration file):

```
python inference.py --CFG config/CFG_vert_bbox_ratio_infer.yaml
```

Prediction generated will saved in the `infer_vert_bbox_ratio.pkl` file at the `./output` directory


## Stage 2:

![Stage 2](docs/Stage2.png)

### Data Preparation

In this step, images used for training will be those with segmentation mask available, which come from 87 patients out of 2019 total number of patients. The labels used in this step include top left coordination `(x0, y0)` and bottom right coordination `(x1, y1)` of the bounding box that cover all vertebrae C1-C7 in each image. The top left coordination `(x0, y0)` can be extracted from segmentation mask by determining the first column from top to bottom and the first row from left to right that contain the element belong to vertebrae C1-C7, while the bottom right coordination `(x1, y1)` can be determined in the similar way except for using the last column and row instead of the first one. The labels also include the ratio of each vertebrae in each image by counting the number of elements belong to each vertebrae in the segmentation mask, then divide by the maximum number of elements belong to each vertebrae in all images of each patient.

To prepare the label file, run `utils/preprocess_data.py --image_dir dataset/train_images --label_path dataset/train.csv --vert_label_path output/infer_vert_bbox_ratio.pkl --get_frac_label --seq_len 24` and the label file `vertebrae_df.pkl` will be generated in the `./output` directory


### Fold Splitting

Next, run `utils/fold_split.py --label_path output/vertebrae_df.pkl` to split data into 5 folds and the new label file `vertebrae_df_fold5.pkl` will be saved in the `./output` directory

```
python utils/fold_split.py

usage: fold_split.py [-h] [--label_path LABEL_PATH] [--kfold KFOLD] [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --label_path LABEL_PATH
                        Path to the label file
  --kfold KFOLD         Number of folds
  --save_path SAVE_PATH
                        Path to the save file
```

### Training

For training, run `train.py` to train model (training customization can be done by modifying the configuration file):

```
python train.py --CFG config/CFG_FD_train.yaml
```

By default, this will run all 5 folds sequentially. To run a single fold instead of all 5 folds, specify `--fold` parameter with a number between 0 and 4

### Inference

For inference, run `inference.py` to generate prediction (inference customization can be done by modifying the configuration file):

```
python inference.py --CFG config/CFG_FD_infer.yaml
```

**Note:** An alternative to run all the commands provided is to run the `train.sh` file

```
sh train.sh
```

<!-- ## Demo

Run `utils/cam_vis.py` to demo classification result with GradCAM:

```
python utils/cam_vis.py
```

## Inference
For inferencing, run `inference.py` to generate predictions on the test set and the output file will be `submission.csv`:

```
python inference.py
```

**Note:** Depend on which checkpoint/model to use for inference, there might be need for modification of checkpoint/model directory inside the `inference.py` file to get the corresponding predictions.  -->