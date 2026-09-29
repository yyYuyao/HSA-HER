import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from transformers import AutoImageProcessor

def pad_or_truncate_tokens(token_array, target_len):
    token_array = np.asarray(token_array, dtype=np.float32)
    if token_array.ndim == 1:
        token_array = token_array[None, :]
    if token_array.ndim != 2:
        raise ValueError(f"Expected retrieval tokens to be 2D, got shape {token_array.shape}")

    seq_len, hidden_dim = token_array.shape
    if seq_len >= target_len:
        return token_array[:target_len]

    padded = np.zeros((target_len, hidden_dim), dtype=np.float32)
    padded[:seq_len] = token_array
    return padded

def normalize_chexpert_rel_path(path):
    path = str(path).replace("\\", "/")
    for prefix in ("chexpert_plus/PNG/", "chexpert_plus/", "PNG/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path

def normalize_mimic_rel_path(path):
    path = str(path).replace("\\", "/")
    for prefix in ("mimic-cxr/images/", "mimic-cxr/mimic-cxr-jpg/files/", "files/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path

def build_mimic_query_key(path):
    return f"mimic-cxr/images/{normalize_mimic_rel_path(path)}"

def resolve_mimic_image_path(base_dir, image_path):
    rel_path = normalize_mimic_rel_path(image_path)
    normalized_base = str(base_dir).replace("\\", "/").rstrip("/")
    candidates = []

    if normalized_base.endswith("mimic-cxr-jpg/files"):
        candidates.append(os.path.join(base_dir, rel_path))
    elif normalized_base.endswith("mimic-cxr"):
        candidates.append(os.path.join(base_dir, "mimic-cxr-jpg", "files", rel_path))
    else:
        candidates.append(os.path.join(base_dir, "mimic-cxr-jpg", "files", rel_path))
        candidates.append(os.path.join(base_dir, "files", rel_path))
        candidates.append(os.path.join(base_dir, rel_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def build_mimic_feature_candidates(root_dir, image_path):
    rel_path = normalize_mimic_rel_path(image_path)
    rel_npy = rel_path.replace(".jpg", ".npy").replace(".png", ".npy")
    raw_npy = str(image_path).replace("\\", "/").replace(".jpg", ".npy").replace(".png", ".npy")

    rel_variants = [
        rel_npy,
        raw_npy,
        build_mimic_query_key(image_path).replace(".jpg", ".npy").replace(".png", ".npy"),
    ]
    candidates = []
    for rel_variant in rel_variants:
        normalized_rel = rel_variant.replace("\\", "/")
        candidates.append(os.path.join(root_dir, normalized_rel))
        candidates.append(os.path.join(root_dir, normalize_mimic_rel_path(normalized_rel)))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        normalized_candidate = os.path.normpath(candidate)
        if normalized_candidate not in seen:
            seen.add(normalized_candidate)
            unique_candidates.append(normalized_candidate)
    return unique_candidates

def build_chexpert_feature_candidates(root_dir, image_path):
    normalized_path = normalize_chexpert_rel_path(image_path)
    npy_rel_path = normalized_path.replace(".jpg", ".npy").replace(".png", ".npy")
    raw_npy_path = str(image_path).replace("\\", "/").replace(".jpg", ".npy").replace(".png", ".npy")

    rel_variants = [npy_rel_path, raw_npy_path]
    candidates = []
    for rel_path in rel_variants:
        normalized_rel = rel_path.replace("\\", "/")
        candidates.append(os.path.join(root_dir, normalized_rel))
        if normalized_rel.startswith("val/"):
            candidates.append(os.path.join(root_dir, "valid/" + normalized_rel[len("val/"):]))
        if normalized_rel.startswith("valid/"):
            candidates.append(os.path.join(root_dir, "val/" + normalized_rel[len("valid/"):]))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        normalized_candidate = os.path.normpath(candidate)
        if normalized_candidate not in seen:
            seen.add(normalized_candidate)
            unique_candidates.append(normalized_candidate)
    return unique_candidates

def summarize_array_health(array):
    array = np.asarray(array)
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    max_abs = float(np.abs(finite_values).max()) if finite_values.size else None
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "posinf_count": int(np.isposinf(array).sum()),
        "neginf_count": int(np.isneginf(array).sum()),
        "max_abs_finite": max_abs,
    }

def ensure_array_is_finite(array, feature_name, sample_id, feature_path, extra_context=None):
    if np.isfinite(array).all():
        return

    stats = summarize_array_health(array)
    context = f", extra={extra_context}" if extra_context else ""
    raise ValueError(
        f"Non-finite values detected in {feature_name} for sample_id={sample_id}, "
        f"path={feature_path}, stats={stats}{context}"
    )

class FieldParser:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(
            args.image_processor
        )

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(
            img,
            return_tensors="pt",
            size={'height': self.args.input_size, 'width': self.args.input_size},
            do_center_crop=False,
        ).pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace("..", ".").replace("..", ".").replace("..", ".").replace("1. ", "") \
                .replace(". 2. ", ". ").replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ") \
                .replace(" 2. ", ". ").replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ") \
                .strip().lower().split(". ")
            sent_cleaner = lambda t: re.sub(
                '[.,?;*!%^&_+():-\\[\\]{}]',
                "",
                t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower(),
            )
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = " . ".join(tokens) + " ."
        elif self.dataset == "chinese":
            pass
        elif self.dataset in ["mimic_cxr", "chexpert_plus"]:
            report_cleaner = lambda t: t.replace("\n", " ").replace("__", "_").replace("__", "_").replace("__", "_") \
                .replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_").replace("  ", " ") \
                .replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ") \
                .replace("..", ".").replace("..", ".").replace("..", ".").replace("..", ".").replace("..", ".") \
                .replace("..", ".").replace("..", ".").replace("..", ".").replace("1. ", "").replace(". 2. ", ". ") \
                .replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ").replace(" 2. ", ". ") \
                .replace(" 3. ", ". ").replace(" 4. ", ". ").replace(" 5. ", ". ").replace(":", " :") \
                .strip().lower().split(". ")
            sent_cleaner = lambda t: re.sub(
                '[.,?;*!%^&_+()\\[\\]{}]',
                "",
                t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower(),
            )
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = " . ".join(tokens) + " ."
        return report

    def parse(self, features):
        if self.dataset == "chinese":
            to_return = {"id": str(features["id"])}
            report = features.get("image_finding", "")
        else:
            to_return = {"id": features["id"]}
            report = features.get("report", "")

        report = self.clean_report(report)
        to_return["input_text"] = report

        images = []
        for image_path in features["image_path"]:
            if self.dataset == "chexpert_plus":
                image_path = normalize_chexpert_rel_path(image_path).replace(".jpg", ".png")
                img_full_path = os.path.join(self.args.base_dir, "PNG", image_path)
            elif self.dataset == "mimic_cxr":
                img_full_path = resolve_mimic_image_path(self.args.base_dir, image_path)
            elif self.dataset == "iu_xray":
                img_full_path = os.path.join(self.args.base_dir, image_path)
            else:
                img_full_path = os.path.join(self.args.base_dir, "mimic-cxr-jpg/files", image_path)

            with Image.open(img_full_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if len(array.shape) != 3 or array.shape[-1] != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)

        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split="train"):
        self.args = args
        self.full_meta = json.loads(open(args.annotation, "r", encoding="utf-8").read())
        self.split = split

        if self.args.drop_unclear_report and split == "train":
            mm = pd.DataFrame(self.full_meta["train"])
            drop_unclear_report_before = len(mm)
            mm = mm[~mm["report"].str.contains("_")]
            mm = mm[mm["report"].apply(lambda x: len(x.split(" ")) > 3)]
            self.full_meta["train"] = mm.to_dict("records")
            drop_unclear_report_after = len(mm)
            print(f"len---drop_unclear_report_before: {drop_unclear_report_before}")
            print(f"len---drop_unclear_report_after: {drop_unclear_report_after}")

        self.train_meta = self.full_meta["train"]

        if self.args.use_feature_mean is False and split == "train" and self.args.dataset in ["mimic_cxr", "chexpert_plus"]:
            mm = pd.DataFrame(self.full_meta["train"])
            if "study_id" not in mm.columns:
                mm["study_id"] = mm["id"].apply(lambda x: "_".join(str(x).split("_")[:3]))
            gb = mm.groupby("study_id")["image_path"].apply(lambda x: sum(x, [])).reset_index()
            self.merged_df = pd.merge(mm, gb, on="study_id", suffixes=("_single", ""))

        self.meta = self.full_meta[split]
        self.parser = FieldParser(args)

        self.retrieval_token_len = getattr(args, "retrieval_token_len", 150)
        self.retrieval_num = getattr(args, "retrieval_num", 5)

        if self.retrieval_num != 5:
            raise ValueError('The paper specifies exactly five retrieved training reports per sample.')
        retrieval_idx_path = args.retrieval_index
        combined_ref_path = args.retrieval_reference_annotation
        self.region_feature_dir = args.region_feature_dir
        self.retrieval_feature_dir = args.retrieval_feature_dir
        for name, value in [('retrieval_index', retrieval_idx_path), ('retrieval_reference_annotation', combined_ref_path), ('region_feature_dir', self.region_feature_dir), ('retrieval_feature_dir', self.retrieval_feature_dir)]:
            if not value:
                raise ValueError(f'--{name} is required for Stage 2 evidence loading.')

        if not os.path.exists(retrieval_idx_path):
            raise FileNotFoundError(f"Missing retrieval index file: {retrieval_idx_path}")
        with open(retrieval_idx_path, "r", encoding="utf-8") as f:
            self.retrieval_index = json.load(f)

        if not os.path.exists(combined_ref_path):
            raise FileNotFoundError(f"Missing reference annotation file: {combined_ref_path}")

        combined_data = json.load(open(combined_ref_path, "r", encoding="utf-8"))
        self.reference_meta = combined_data["train"]
        print(f"Loaded {self.args.dataset} reference bank with {len(self.reference_meta)} samples.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        if self.args.use_feature_mean is False and self.split == "train" and self.args.dataset in ["mimic_cxr", "chexpert_plus"]:
            img_path_num = len(self.merged_df.iloc[index]["image_path"])
            if img_path_num == 2:
                self.meta[index]["image_path"] = self.merged_df.iloc[index]["image_path"]
            elif img_path_num > 2:
                self.meta[index]["image_path"] = self.meta[index]["image_path"] + [random.choice(self.merged_df.iloc[index]["image_path"])]
            else:
                self.meta[index]["image_path"] = self.meta[index]["image_path"] + self.meta[index]["image_path"]

        parsed_data = self.parser.transform_with_parse(self.meta[index])
        ann = self.meta[index]
        image_path = ann["image_path"][0]

        if self.args.dataset == "chexpert_plus":
            image_path = normalize_chexpert_rel_path(image_path).replace(".jpg", ".png")
        elif self.args.dataset == "mimic_cxr":
            image_path = normalize_mimic_rel_path(image_path)

        if "labels" in ann:
            cls_labels = torch.from_numpy(np.array(ann["labels"])).long()[:14]
        else:
            if self.split != 'test':
                raise KeyError(f"Disease labels are missing for Stage 2 sample {ann.get('id', 'Unknown')}")
            cls_labels = torch.zeros(14, dtype=torch.long)
        parsed_data["cls_labels"] = cls_labels

        original_img_path = ann["image_path"][0]
        if self.args.dataset == "chexpert_plus":
            json_key = f"chexpert_plus/PNG/{normalize_chexpert_rel_path(original_img_path)}"
        elif self.args.dataset == "iu_xray":
            json_key = original_img_path
        elif self.args.dataset == "mimic_cxr":
            json_key = build_mimic_query_key(original_img_path)
        else:
            json_key = original_img_path if original_img_path.startswith("mimic-cxr") else f"mimic-cxr/images/{original_img_path}"

        if json_key not in self.retrieval_index:
            raise KeyError(f'No offline retrieval entry for {json_key}')
        clip_indices = self.retrieval_index[json_key]
        if not isinstance(clip_indices, list) or len(clip_indices) < self.retrieval_num:
            raise ValueError(f'Expected at least {self.retrieval_num} retrieved training indices for {json_key}')
        clip_indices = clip_indices[:self.retrieval_num]
        if len(set(clip_indices)) != self.retrieval_num:
            raise ValueError(f'Retrieval entry for {json_key} repeats a training reference')
        if any(type(idx) is not int or idx < 0 or idx >= len(self.reference_meta) for idx in clip_indices):
            raise ValueError(f'Retrieval index for {json_key} contains a non-training reference position')

        if self.args.dataset == "mimic_cxr":
            mimic_region_roots = [
                self.region_feature_dir,
            ]
            region_candidates = []
            for root_dir in mimic_region_roots:
                region_candidates.extend(build_mimic_feature_candidates(root_dir, ann["image_path"][0]))
            rt_path = next((path for path in region_candidates if os.path.exists(path)), region_candidates[0])
        elif self.args.dataset == "iu_xray":
            rt_path = os.path.join(self.region_feature_dir, image_path).replace(".png", ".npy").replace(".jpg", ".npy")
        else:
            region_candidates = build_chexpert_feature_candidates(
                self.region_feature_dir,
                image_path,
            )
            rt_path = next((path for path in region_candidates if os.path.exists(path)), region_candidates[0])

        if not os.path.exists(rt_path):
            raise FileNotFoundError(f"Region TXT file not found: {rt_path}")

        region_array = np.load(rt_path)
        ensure_array_is_finite(region_array, "region_txt", ann.get("id", "unknown"), rt_path)
        region_txt = torch.from_numpy(region_array).to(dtype=torch.float32)
        parsed_data["region_txt"] = region_txt
        parsed_data["region_txt_path"] = rt_path

        retrieval_txts = []
        retrieval_sources = []
        for dsrf_idx in clip_indices:
            ref_ann = self.reference_meta[dsrf_idx]
            ref_img_path = ref_ann["image_path"][0]

            if self.args.dataset == 'iu_xray':
                txt_path = ref_img_path.replace('.jpg', '.npy').replace('.png', '.npy')
                candidate_search_paths = [os.path.join(self.retrieval_feature_dir, txt_path)]
            elif self.args.dataset == 'mimic_cxr':
                candidate_search_paths = build_mimic_feature_candidates(self.retrieval_feature_dir, ref_img_path)
            else:
                candidate_search_paths = build_chexpert_feature_candidates(self.retrieval_feature_dir, ref_img_path)

            existing_path = next((path for path in candidate_search_paths if os.path.exists(path)), None)
            if existing_path is None:
                raise FileNotFoundError(f'Retrieved training report embedding is missing for index {dsrf_idx}: {candidate_search_paths}')
            loaded_array = np.load(existing_path)
            ensure_array_is_finite(loaded_array, 'retrieval_txt', ann.get('id', 'unknown'), existing_path, extra_context={'retrieved_from': ref_img_path, 'retrieval_index': dsrf_idx})
            txt_emb = pad_or_truncate_tokens(loaded_array, self.retrieval_token_len)
            retrieval_txts.append(torch.from_numpy(txt_emb).to(dtype=torch.float32))
            retrieval_sources.append(existing_path)

        parsed_data["retrieval_txt"] = torch.stack(retrieval_txts, dim=0)
        parsed_data["retrieval_txt_sources"] = " | ".join(retrieval_sources)
        return parsed_data

def create_datasets(args):
    return ParseDataset(args, "train"), ParseDataset(args, "val"), ParseDataset(args, "test")
