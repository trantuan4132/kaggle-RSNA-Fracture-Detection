python data_preprocessing.py \
    --train_image_dir "/content/images_dir" \
    --metadata "/content/df_data.csv" \
    --bounding_boxes "/content/onedrive/Dataset/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv" \
    --yolo_dir "/content/kaggle-RSNA-Fracture-Detection/yolov5-detector/yolov5" \
    --train True 
python train_hyperparams.py
cd ./yolov5
python train.py --img 1024 --batch 8 --epochs 2 --data my_data.yaml --cfg './models/yolov5l.yaml' --weights yolo_model_path