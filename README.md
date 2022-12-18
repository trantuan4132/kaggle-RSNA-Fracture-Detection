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

The overall dataset structure will be as follow:

```
dataset
├── segmentations
│   ├── 1.2.826.0.1.3680043.10633.nii
│   ├── 1.2.826.0.1.3680043.10921.nii
│   └── ...
│
├── train_images
│   ├── 1.2.826.0.1.3680043.10001
│   │   ├── 1.dcm
│   │   ├── 2.dcm
│   │   └── ...
│   ├── 1.2.826.0.1.3680043.10005
│   │   ├── 1.dcm
│   │   ├── 2.dcm
│   │   └── ...
│   └── ...
│
├── test_images
│   ├── 1.2.826.0.1.3680043.22327
│   │   ├── 1.dcm
│   │   ├── 2.dcm
│   │   └── ...
│   ├── 1.2.826.0.1.3680043.25399
│   │   ├── 1.dcm
│   │   ├── 2.dcm
│   │   └── ...
│   └── ...
│
├── train.csv
├── train_bounding_boxes.csv
├── test.csv
└── sample_submission.csv
```

## Stage 1: Vertebrae localization and ratio regression

![Stage 1](docs/Stage1.png)

### Data Preparation

In this step, images used for training will be those with segmentation mask available, which come from 87 patients out of 2019 total number of patients. The labels used in this step include top left coordination `(x0, y0)` and bottom right coordination `(x1, y1)` of the bounding box that cover all vertebrae C1-C7 in each image. The top left coordination `(x0, y0)` can be extracted from segmentation mask by determining the first column from top to bottom and the first row from left to right that contain the element belong to vertebrae C1-C7, while the bottom right coordination `(x1, y1)` can be determined in the similar way except for using the last column and row instead of the first one. The labels also include the ratio of each vertebrae in each image by counting the number of elements belong to each vertebrae in the segmentation mask, then divide by the maximum number of elements belong to each vertebrae in all images of each patient.

In summary, each row in the label file contains:
- $\begin{matrix}
x0 = \min_{N_i > 0} i \\
y0 = \min_{N_j > 0} j \\
x1 = \max_{N_i > 0} i \\
y1 = \max_{N_j > 0} j
\end{matrix}
\left\{
  \begin{matrix}
    \begin{aligned}
      &N_i: \text{number of elements belong to vertebrae C1-C7 in the } i^{th} \text{ column of the slice } (i = 1 \rightarrow n_{col}) \\ 
      &N_j: \text{number of elements belong to vertebrae C1-C7 in the } j^{th} \text{ row of the slice } (j = 1 \rightarrow n_{row}) \\ 
      &x0, y0: \text{top left coordination of the bounding box} \\
      &x1, y1: \text{bottom right coordination of the bounding box}
    \end{aligned}
  \end{matrix}
\right.$
- $R^k_{Ci} = \frac{N^k_{Ci}}{\max(N^1_{Ci}, N^2_{Ci}, ..., N^T_{Ci})} 
\left\{
  \begin{matrix}
    \begin{aligned}
      &T: \text{total number of slices in a study}
      \\ 
      &N^k_{Ci}: \text{number of elements belong to vertebrae } Ci \text{ in the } k^{th} \text{ slice } (i = 1 \rightarrow 7, k = 1 \rightarrow T)
      \\ 
      &R^k_{Ci}: \text{ratio of each vertebrae } Ci \text{ in the } k^{th} \text{ slice }
    \end{aligned}
  \end{matrix}
\right.$

To prepare the label file, run `python utils/preprocess_data.py --image_dir dataset/train_images --label_path dataset/train.csv --seg_dir dataset/segmentations --get_vert_label --get_vert_bbox` and the label file `train_vert_bbox_ratio.csv` will be generated in the `./output` directory

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

Next, run `python utils/fold_split.py --label_path output/train_vert_bbox_ratio.csv --multi_label` to split data into 5 folds using GroupKFold with *StudyInstanceUID* as group and the new label file `train_vert_bbox_ratio_fold5.csv` will be saved in the `./output` directory

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

For training, use `train.py` to train model (training customization can be done by modifying the configuration file):

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

For inference, use `inference.py` to generate prediction on all 2019 patients using an ensemble of 5 models, each of which is trained on 4 out of 5 folds and evaluated on the remaining fold (inference customization can be done by modifying the configuration file):

```
python inference.py --CFG config/CFG_vert_bbox_ratio_infer.yaml
```

The prediction generated on each slice will be further processed by determining only one bounding box for each study by getting the minimum of `x0`, `y0` and the maximum of `x1`, `y1` across all slices with the ratio of any vertebrae larger than 0.5 in each study. Moreover, for each vertebrae in a study, gather all slices in that study in which the ratio of that vertebrae is larger than 0.5 into a single list. After that, the final prediction will have a unique study id in each row along with a bounding box's top left and bottom right coordination, and each vertebrae in each column with a list of slices belonging to it as value. The final prediction will be saved in the `infer_vert_bbox_ratio.pkl` file in the `./output` directory.


## Stage 2: Fracture Detection

![Stage 2](docs/Stage2.png)

### Data Preparation

In this stage, the images used for training come from all of the 2019 studies except for a study with the id `1.2.826.0.1.3680043.20574` since this study does not contain any slices belonging to C1-C7. Each fracture label from `train.csv` will then be assigned to a list of slices belonging to each vertebrae in each study and a fixed number of slices (i.e. 24) will be chosen from that list of slices using evenly spaced indices (e.g. 47 slices -> 24 slices with index 0, 2, 4, ..., 46).

To prepare the label file, run `python utils/preprocess_data.py --image_dir dataset/train_images --label_path dataset/train.csv --vert_label_path output/infer_vert_bbox_ratio.pkl --get_frac_label --seq_len 24` and the label file `vertebrae_df.pkl` will be generated in the `./output` directory


### Fold Splitting

Next, run `python utils/fold_split.py --label_path output/vertebrae_df.pkl` to split data into 5 folds using GroupKFold with *StudyInstanceUID* as group and the new label file `vertebrae_df_fold5.pkl` will be saved in the `./output` directory

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

For training, use `train.py` to train model (training customization can be done by modifying the configuration file):

```
python train.py --CFG config/CFG_FD_train.yaml
```

By default, this will run all 5 folds sequentially. To run a single fold instead of all 5 folds, specify `--fold` parameter with a number between 0 and 4

### Inference

For inference, use `inference.py` to generate prediction (inference customization can be done by modifying the configuration file):

```
python inference.py --CFG config/CFG_FD_infer.yaml
```

**Note:** An alternative to run all the commands provided is to run the `train.sh` file

```
sh train.sh
```

## Demo

https://user-images.githubusercontent.com/65863754/208296029-7f59db4a-13c3-47ce-8ce3-c407897ecb44.mp4

<!-- Run `utils/cam_vis.py` to demo classification result with GradCAM:

```
python utils/cam_vis.py
```

## Inference
For inferencing, run `inference.py` to generate predictions on the test set and the output file will be `submission.csv`:

```
python inference.py
```

**Note:** Depend on which checkpoint/model to use for inference, there might be need for modification of checkpoint/model directory inside the `inference.py` file to get the corresponding predictions.  -->
