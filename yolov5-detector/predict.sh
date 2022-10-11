TEST_IMAGE_DIR = '/content/drive/MyDrive/Work/FPT_Capstone_YOLOv5/data/val/fracture/'
WEIGHTS = '/content/drive/MyDrive/Work/FPT_Capstone_YOLOv5/weights/best.pt'
IMG_SIZE = 512
CSV_SAVE_DIR = './'
YOLO_PREDICT_DIR = '/content/yolov5/runs/predict-cls/exp2/labels'
python classify/predict.py --source TEST_IMAGE_DIR --weights WEIGHTS --imgsz IMG_SIZE
cd ..
python get_results_csv.py --yolo-predict-dir  --save-dir CSV_SAVE_DIR