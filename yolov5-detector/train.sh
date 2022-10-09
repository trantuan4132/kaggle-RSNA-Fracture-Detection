pip install wandb
WANDB_API_KEY="250c23fe6f3b5730d50ba78a9c8ed669ed732f0e"
WANDB_ENTITY="aip490"
WANDB_PROJECT="RSNA-YOLO-DETECT"
wandb login
python data_preprocessing.py \
    --train_image_dir "/content/images_dir" \
    --metadata "/content/df_data.csv" \
    --bounding_boxes "/content/onedrive/Dataset/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv" \
    --yolo_dir "/content/drive/MyDrive/FPT_Capstone_Project/kaggle-RSNA-Fracture-Detection/yolov5-detector/yolov5" 
python train_hyperparams.py
cd ./yolov5
python classify/train.py --img 512 --batch 32 --epochs 80 --model "yolov5l-cls.pt" --data "/content/drive/MyDrive/FPT_Capstone_Project/kaggle-RSNA-Fracture-Detection/yolov5-detector/yolov5/base_dir/images" 