#!/bin/bash

dataset="iu_xray"
annotation="/path/to/annotation.json"
base_dir="/path/to/images"
clip_stage_ckpt="${CLIP_STAGE_CKPT:?Set CLIP_STAGE_CKPT}"
if [ $# -eq 0 ]; then
    echo "If you need to test the model, please provide the absolute path of the model to be tested after training with this script."
fi
load_model=$1
if [[ $load_model != "" ]]; then
    echo "++++++++test++++++++"
    savepath="${load_model}_test"
    echo "The running files will be saved to $savepath"
    test_mode="--test \
              --beam_size 5\
              --ckpt_file ${load_model} \
              --test_batch_size 32 \
              "
else
    echo "++++++++train++++++++"
    script_name=$(basename "$0" .sh)
    savepath="./save/$dataset/$script_name"
    echo "The running files will be saved to $savepath"
    test_mode=""
fi
mkdir -p "$savepath"
script_name=$(basename "$0")
script_dir=$(dirname "$0")
source_script="${script_dir}/${script_name}"
cp "$source_script" "${savepath}"
CUDA_VISIBLE_DEVICES=0 python train_hsa_her_stage2.py \
     $test_mode \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --context_pair_seed 4096 \
    --beam_size 5 \
    --chosen vmamba \
    --llm qwen \
    --her_factory "${HER_FACTORY:?Set HER_FACTORY=module:factory}" \
    --chexbert_path "${CHEXBERT_PATH:?Set CHEXBERT_PATH}" \
    --vision_model "${VISION_MODEL:?Set VISION_MODEL}" \
    --llama_model "${LLM_MODEL:-Qwen/Qwen1.5-1.8B-Chat}" \
    --batch_size 8  \
    --val_batch_size 24 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --accumulate_grad_batches 1 \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 25 \
    --limit_val_batches 1 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 0 \
    --input-size 224 \
    --strategy deepspeed \
    --cls_weight 4.0 \
    --retrieval_num 5 \
    --retrieval_token_len 150 \
    --num_img_self_exp 4 \
    --num_img_local_exp 4 \
    --num_img_global_exp 4 \
    --clip_stage_ckpt "${clip_stage_ckpt}" \
    --retrieval_index "${RETRIEVAL_INDEX:?Set RETRIEVAL_INDEX}" \
    --retrieval_reference_annotation "${RETRIEVAL_REFERENCE_ANNOTATION:?Set RETRIEVAL_REFERENCE_ANNOTATION}" \
    --region_feature_dir "${REGION_FEATURE_DIR:?Set REGION_FEATURE_DIR}" \
    --retrieval_feature_dir "${RETRIEVAL_FEATURE_DIR:?Set RETRIEVAL_FEATURE_DIR}" \
    2>&1 |tee -a ${savepath}/log.txt
