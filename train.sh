python utils/process_data.py \
    --image_dir "train_images" \
    --label_path "train.csv" \
    --relabel_path "relabel.csv" \
    --seg_dir "segmentations" \
    --out_dir "." \
    --drop \
    --get_rvs_lst \
    --get_vert_label
python utils/fold_split.py \
    --label_path "train.csv" \
    --kfold 5 \
    --save_path "train_fold5.csv"
# python utils/fold_split.py \
#     --label_path "train_segmented.csv" \
#     --kfold 5 \
#     --save_path "train_segmented_fold5.csv"
python utils/preprocess_checkpoint.py \
    --checkpoint_path "convnext_tiny_22k_1k_384.pth" \
    --model "convnext" \
    --variant "convnext_tiny"
python train_single_baseline.py