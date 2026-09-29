#!/bin/bash
set -e

RESULT_DIR="/path/to/result"

REFS_JSON="${RESULT_DIR}/test_refs.json"
HYPS_JSON="${RESULT_DIR}/test_result.json"

OUTPUT_DIR="/path/to/mimic_cxr"
FILE_NAME="hsa_her_eval"

MODEL_NAME="/path/to/GREEN-radllama2-7b"

TEST_DIR_NAME=$(basename $(dirname "$RESULT_DIR"))

LOG_FILE="${OUTPUT_DIR}/green_log_${TEST_DIR_NAME}.txt"

mkdir -p "${OUTPUT_DIR}"

if [ "$1" != "bg" ]; then
    echo "🚀 正在将任务提交到后台运行..."

    SCRIPT_PATH=$(readlink -f "$0")

    nohup bash "$SCRIPT_PATH" bg > "$LOG_FILE" 2>&1 &
    echo "✅ 任务已成功在后台启动！"
    echo "📄 所有的输出日志将保存至: $LOG_FILE"
    echo "🔍 你可以随时输入此命令实时查看进度: tail -f $LOG_FILE"
    echo "👋 现在你可以安全地关闭当前终端了。"
    exit 0
fi

echo "========================================"
echo "🚀 开始进行 GREEN 离线评测"
echo "参考报告: ${REFS_JSON}"
echo "生成报告: ${HYPS_JSON}"
echo "输出目录: ${OUTPUT_DIR}"
echo "========================================"

if [ ! -f "$REFS_JSON" ] || [ ! -f "$HYPS_JSON" ]; then
    echo "错误: 找不到指定的 json 文件，请确认模型测试已完成且路径正确。"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=3 python -m green_score.green \
    "${REFS_JSON}" \
    "${HYPS_JSON}" \
    "${OUTPUT_DIR}" \
    "${FILE_NAME}" \
    "${MODEL_NAME}"

echo "✅ GREEN 基础评测完成！结果保存在: ${OUTPUT_DIR}/results_${FILE_NAME}.csv"
echo "📊 正在自动计算均值并生成汇总表格..."

python -c "
import pandas as pd
import os

csv_path = '${OUTPUT_DIR}/results_${FILE_NAME}.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    cols = [c for c in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', 'Matched Findings', 'GREEN'] if c in df.columns]
    mean_df = df[cols].mean().to_frame().T

    error_cols = [c for c in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'] if c in cols]
    if error_cols:
        mean_df['Sum_Error'] = mean_df[error_cols].sum(axis=1)

    out_path = os.path.join('${OUTPUT_DIR}', 'summary_${FILE_NAME}.csv')
    mean_df.to_csv(out_path, index=False)
    print(f'🎉 汇总计算完成！可以直接填入论文表格的数据已保存至: {out_path}')
"
