#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

dataset="${DATASET:?Set DATASET to chexpert_plus, iu_xray, or mimic_cxr}"
annotation="/path/to/annotation.json"
base_dir="/path/to/data/"

version="HSA_HER_stage1"
savepath="save/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

nohup python -u train_hsa_her_stage1.py \
    --data_mode clip \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 128  \
    --chosen vmamba \
    --vision_model "${VISION_MODEL:?Set VISION_MODEL}" \
    --freeze_vm False \
    --vis_use_lora False \
    --use_itc True \
    --use_cls True \
    --use_homo True \
    --itc_weight 1.0 \
    --clip_cls_weight 1.0 \
    --image_cls_weight 1.0 \
    --homo_weight 1.0 \
    --savedmodel_path ${savepath} \
    --num_workers 32 \
    --devices 4 \
    --max_epochs 10 \
    --temperature 0.05 \
    --strategy deepspeed \
    --alignment_factory "${ALIGNMENT_FACTORY:?Set ALIGNMENT_FACTORY=module:factory}" \
    >> ${savepath}/log.txt 2>&1 &

