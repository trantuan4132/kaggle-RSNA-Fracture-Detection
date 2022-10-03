pip install wandb
WANDB_API_KEY="250c23fe6f3b5730d50ba78a9c8ed669ed732f0e"
WANDB_ENTITY="aip490"
WANDB_PROJECT="RSNA-YOLO-DETECT"
wandb login
python data_preprocessing.py \
    --train_image_dir "/content/images_dir" \
    --metadata "/content/df_data.csv" \
    --bounding_boxes "/content/onedrive/Dataset/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv" \
    --yolo_dir "/content/kaggle-RSNA-Fracture-Detection/yolov5-detector/yolov5" 
python train_hyperparams.py
cd ./yolov5
python train.py --img 1024 --batch 8 --epochs 2 --data my_data.yaml --cfg './models/yolov5l.yaml' --weights "/content/drive/MyDrive/AIP490_Capstone Project/Dataset/PNG data/yolov5l.pt"