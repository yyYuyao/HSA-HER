dataset="mimic_cxr"
annotation="/path/to/annotation.json"
base_dir="/path/to/images"
clip_stage_ckpt="${CLIP_STAGE_CKPT:?Set CLIP_STAGE_CKPT}"
if [ $# -eq 0 ]; then
    echo "If you need to test the model, please provide the absolute path of the model to be tested after training with this script."
fi
load_model=$1
if [[ $load_model != "" ]]; then
    echo "++++++++test++++++++"

    script_name=$(basename "$0" .sh)
    current_time=$(date -u -d '+8 hours' "+%Y%m%d_%H%M%S")

    savepath="/path/to/${script_name}_test_${current_time}"

    echo "The running files will be saved to $savepath"
    test_mode="--test \
              --beam_size 3 \
              --ckpt_file ${load_model} \
              --test_batch_size 24 \
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
CUDA_VISIBLE_DEVICES=3 nohup python train_hsa_her_stage2.py \
    $test_mode \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --chosen vmamba \
    --llm llama2 \
    --her_factory "${HER_FACTORY:?Set HER_FACTORY=module:factory}" \
    --chexbert_path "${CHEXBERT_PATH:?Set CHEXBERT_PATH}" \
    --vision_model "${VISION_MODEL:?Set VISION_MODEL}" \
    --llama_model "${LLM_MODEL:-meta-llama/Llama-2-7b-chat-hf}" \
    --batch_size 8  \
    --val_batch_size 24 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --accumulate_grad_batches 1 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 1 \
    --max_epochs 6 \
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
    >> ${savepath}/log.txt 2>&1 &
