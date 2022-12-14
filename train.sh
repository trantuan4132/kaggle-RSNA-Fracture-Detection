python utils/process_data.py \
    --image_dir "train_images" \
    --label_path "train.csv" \
    --relabel_path "relabel.csv" \
    --seg_dir "segmentations" \
    --out_dir "dataset" \
    --drop \
    --get_rvs_lst \
    --get_vert_label \
    --get_vert_bbox

python utils/fold_split.py \
    --label_path "train_vert_bbox_ratio.csv" \
    --multi_label \
    --kfold 5 \
    --save_path "train_vert_bbox_ratio_fold5.csv"

# python utils/preprocess_checkpoint.py \
#     --checkpoint_path "convnext_tiny_22k_1k_384.pth" \
#     --model "convnext" \
#     --variant "convnext_tiny"

for FOLD in 0 1 2 3 4
do
    python train.py --CFG config/CFG_vert_bbox_ratio_train.yaml --fold $FOLD
done

python inference.py --CFG config/CFG_vert_bbox_ratio_infer.yaml

python utils/process_data.py \
    --image_dir "train_images" \
    --label_path "train.csv" \
    --vert_label_path "infer_vert_bbox_ratio.pkl"
    --out_dir "dataset" \
    --get_frac_label \
    --seq_len 24

python fold_split.py \
    --label_path 'vertebrae_df.pkl' \
    --label_cols 'fractured' \
    --kfold 5 \
    --save_path 'vertebrae_df_fold5.pkl'

for FOLD in 0 1 2 3 4
do
    python train.py --CFG config/CFG_FD_train.yaml --fold $FOLD
done

python inference.py --CFG config/CFG_vert_bbox_FD_infer.yaml