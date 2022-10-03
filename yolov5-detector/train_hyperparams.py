import yaml

if __name__ == "__main__":
    yaml_dict = {  
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.032,  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay 5e-4
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
        'box': 0.1,  # box loss gain
        'cls': 1.0,  # cls loss gain
        'cls_pw': 0.5,  # cls BCELoss positive_weight
        'obj': 2.0,  # obj loss gain (scale with pixels)
        'obj_pw': 0.5,  # obj BCELoss positive_weight
        'iou_t': 0.20,  # IoU training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        'anchors': 0,  # anchors per output layer (0 to ignore)
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        'hsv_h': 0,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0,  # image HSV-Value augmentation (fraction)
        'degrees': 30.0,  # image rotation (+/- deg)
        'translate': 0.2,  # image translation (+/- fraction)
        'scale': 0.3,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.2,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 0.8,  # image mosaic (probability)
        'mixup': 0.0  # image mixup (probability) 
    }

    with open(r'yolov5/my_hyp.yaml', 'w') as file:
        documents = yaml.dump(yaml_dict, file)
