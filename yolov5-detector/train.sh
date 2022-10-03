python data_processing.py \
    --train_image_dir "train_images" \
    --metadata "metadata.csv" \
    --bounding_boxes "bounding_boxes.csv" \
    --yolo_dir "yolov5" \
    --out_dir "." \
    --train True \
python train_hyperparams.py
cd yolov5
python train.py --img 1024 --batch 8 --epochs 2 --data my_data.yaml --cfg './models/yolov5l.yaml' --weights yolo_model_path