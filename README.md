

## Release Status

- [x] Stage 1 and Stage 2 training and evaluation entry points.
- [x] Data loading interfaces, configuration templates, and evaluation metrics.
- [x] VMamba and metric dependencies with third-party notices.
- [ ] Pretrained checkpoints, datasets, annotations, and precomputed evidence are not included.

The HSA alignment loss and HER expert and gating implementations are not included in this code release. The feature extraction and retrieval-index construction scripts are also not included. The external modules must be supplied through `--alignment_factory` and `--her_factory`.

## Getting Started

### 1. Install Environment

Create a Python environment with a PyTorch and CUDA combination supported by your system. Install the VMamba requirements and the Python packages imported by the public training code:

```bash
pip install -r VMamba/requirements.txt
pip install lightning transformers peft einops pandas numpy pillow sentencepiece
```

Build the selective-scan CUDA extension according to `VMamba/kernels/selective_scan/README.md`. Java is required by the included METEOR and tokenization tools. A complete pinned environment is not supplied in this partial release.

### 2. Data Preparation

The code has interfaces for **CheXpert Plus**, **IU X-Ray**, and **MIMIC-CXR**. Supply images and an annotation JSON with `train`, `val`, and `test` lists. The datasets, splits, labels, weights, and evidence files are not distributed here.

For Stage 2, provide:

- `--retrieval_index`: offline top-5 retrieval indices for each image.
- `--retrieval_reference_annotation`: annotation JSON whose `train` list is the retrieval reference bank.
- `--region_feature_dir`: precomputed local entity feature files.
- `--retrieval_feature_dir`: precomputed text features of retrieved reports.
- `--chexbert_path`: CheXbert checkpoint for clinical evaluation.

The loader reads five distinct entries from the supplied training reference list. This repository does not contain the feature or index generation code, so their provenance must be checked separately.

### 3. Directory Structure

```text
HSA-HER/
├── configs/              # Command-line configuration and class priors
├── dataset/              # Stage 1 and Stage 2 data loading
├── models/               # Public model wrappers and external-module interfaces
├── scripts/              # Training and evaluation command templates
├── evalcap/              # NLG and CheXbert metrics
├── green_score/          # GREEN evaluation
├── VMamba/               # Visual backbone and CUDA extension source
├── train_hsa_her_stage1.py
└── train_hsa_her_stage2.py
```

## Training

Replace the placeholder data and checkpoint paths in the shell scripts before running them. Stage 1 requires a VMamba checkpoint and an external HSA loss module:

```bash
export DATASET=mimic_cxr
export VISION_MODEL=/path/to/vmamba_checkpoint.pth
export ALIGNMENT_FACTORY=your_module:your_factory
bash scripts/hsa_her_stage1.sh
```

Stage 2 requires a Stage 1 checkpoint, the external HER module, a language model, CheXbert, and the evidence files described above. Use the script for the selected dataset:

```bash
export CLIP_STAGE_CKPT=/path/to/stage1_checkpoint.pth
export HER_FACTORY=your_module:your_factory
export CHEXBERT_PATH=/path/to/chexbert_checkpoint.pth
export RETRIEVAL_INDEX=/path/to/retrieval_index.json
export RETRIEVAL_REFERENCE_ANNOTATION=/path/to/annotation.json
export REGION_FEATURE_DIR=/path/to/region_features
export RETRIEVAL_FEATURE_DIR=/path/to/report_features
bash scripts/hsa_her_mimic_cxr.sh
```

Stage 2 selects the checkpoint with the highest `0.5 × BLEU-4 + 0.5 × CIDEr` score on the validation split. The IU X-Ray script uses Qwen1.5-1.8B-Chat; the CheXpert Plus and MIMIC-CXR scripts use Llama-2-7B-Chat.

## Evaluation

Pass the selected Stage 2 checkpoint to the corresponding dataset script for independent test evaluation:

```bash
bash scripts/hsa_her_mimic_cxr.sh /path/to/best_stage2_checkpoint.pth
```

The public evaluation code reports BLEU-1 to BLEU-4, ROUGE-L, CIDEr, and CheXbert-based clinical metrics. Separate scripts under `scripts/` run GREEN evaluation from saved predictions and references.

## License

No project-wide license is specified in this release. Third-party source and notices are documented in `THIRD_PARTY_NOTICES.txt`.
