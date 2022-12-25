INPUT_DIR="dataset"
OUTPUT_DIR="output"

python3 inference.py \
    --CFG config/CFG_vert_bbox_ratio_infer.yaml \
    --label_file "${OUTPUT_DIR}/input_stage_1.csv" \
    --image_dir "${INPUT_DIR}/train_images" \
    --image_format "jpg" \
    --out_file "${OUTPUT_DIR}/infer_stage_1.pkl"

python3 utils/process_data.py \
    --image_dir "${INPUT_DIR}/train_images" \
    --label_path "${INPUT_DIR}/train.csv" \
    --vert_label_path "${OUTPUT_DIR}/infer_stage_1.pkl" \
    --out_dir $OUTPUT_DIR \
    --out_file "input_stage_2.pkl" \
    --get_frac_label \
    --vert_thresh 0.3 \
    --seq_len 24

python3 inference.py \
    --CFG config/CFG_FD_infer.yaml \
    --label_file "${OUTPUT_DIR}/input_stage_2.pkl" \
    --image_dir "${INPUT_DIR}/train_images" \
    --image_format "jpg" \
    --out_file "${OUTPUT_DIR}/infer_stage_2.csv"