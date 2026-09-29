import os
import json
import re
import copy
import random

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from transformers import AutoImageProcessor
from dataset.hsa_her_dataset import normalize_chexpert_rel_path, resolve_mimic_image_path

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
            do_center_crop=False
        ).pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        elif self.dataset == "chinese":
            None
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'

        return report

    def parse(self, features):
        if self.dataset == "chinese":
            to_return = {'id': str(features['id'])}
            report = features.get("image_finding", "")
        else:
            to_return = {'id': features['id']}
            report = features.get("report", "")

        report = self.clean_report(report)
        to_return['input_text'] = report

        images = []
        for image_path in features['image_path']:
            if self.dataset == 'chexpert_plus':
                rel_path = normalize_chexpert_rel_path(image_path).replace('.jpg', '.png')
                full_path = os.path.join(self.args.base_dir, 'PNG', rel_path)
            elif self.dataset == 'mimic_cxr':
                full_path = resolve_mimic_image_path(self.args.base_dir, image_path)
            else:
                full_path = os.path.join(self.args.base_dir, image_path)
            if not os.path.exists(full_path) and full_path.endswith('.jpg'):
                png_path = full_path[:-4] + '.png'
                if os.path.exists(png_path):
                    full_path = png_path

            with Image.open(full_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)

        to_return["image"] = images
        return to_return

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split

        self.meta = json.loads(open(args.annotation, 'r', encoding='utf-8').read())

        if getattr(self.args, 'drop_unclear_report', False) and split == 'train':
            meta_frame = pd.DataFrame(self.meta['train'])
            if 'report' in meta_frame.columns:
                meta_frame = meta_frame[~meta_frame['report'].str.contains('_', na=False)]
                meta_frame = meta_frame[meta_frame['report'].apply(lambda x: len(str(x).split(' '))) > 3]
            self.meta['train'] = meta_frame.to_dict('records')

        if getattr(self.args, 'use_feature_mean', False) is False and split == 'train':
            meta_frame = pd.DataFrame(self.meta['train'])
            grouped = meta_frame.groupby(['subject_id', 'study_id'])['image_path'].apply(lambda x: sum(x, [])).reset_index()
            self.merged_df = pd.merge(meta_frame, grouped, on=['subject_id', 'study_id'], suffixes=("_single", ""))

        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        sample_meta = copy.deepcopy(self.meta[index])

        if getattr(self.args, 'use_feature_mean', False) is False and self.split == 'train':
            img_path_num = len(self.merged_df.iloc[index]['image_path'])
            if img_path_num == 2:
                sample_meta['image_path'] = self.merged_df.iloc[index]['image_path']
            elif img_path_num > 2:
                sample_meta['image_path'] = sample_meta['image_path'] + [random.choice(self.merged_df.iloc[index]['image_path'])]
            else:
                sample_meta['image_path'] = sample_meta['image_path'] + sample_meta['image_path']

        base_dict = self.parser.parse(sample_meta)

        raw_labels = sample_meta.get('labels', sample_meta.get('label'))
        if raw_labels is None:
            if self.split != 'test':
                raise KeyError(f"Disease labels are missing for Stage 1 sample {sample_meta['id']}")
            raw_labels = [0] * 14
        if len(raw_labels) != 14:
            raise ValueError(f"Expected 14 disease labels for sample {sample_meta['id']}")
        labels = torch.tensor(raw_labels, dtype=torch.long)

        base_dict['labels'] = labels
        return base_dict

def create_datasets(args):
    return ParseDataset(args, "train"), ParseDataset(args, "val"), ParseDataset(args, "test")
