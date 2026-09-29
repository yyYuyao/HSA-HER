import os
import torch
import numpy as np
from .chexbert import CheXbert

"""
0 = blank
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity',
    'lung_lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis',
    'pneumothorax', 'pleural_effusion', 'pleural_other', 'fracture',
    'support_devices', 'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device, bert_model_path='bert-base-uncased'):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.bert_model_path = bert_model_path
        self.chexbert = CheXbert(self.checkpoint_path, self.device, self.bert_model_path).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)], res[i:min(i + mbatch_size, length)]

    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []


        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert

        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)


        res_bin = (res_chexbert == 1)
        gts_bin = (gts_chexbert == 1)


        tp = (res_bin & gts_bin).astype(float)
        fp = (res_bin & ~gts_bin).astype(float)
        fn = (~res_bin & gts_bin).astype(float)


        tp_eg = tp.sum(axis=1)
        fp_eg = fp.sum(axis=1)
        fn_eg = fn.sum(axis=1)

        ce_precision = np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean()
        ce_recall = np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean()
        ce_f1 = np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean()


        tp_cls = tp.sum(axis=0)
        fp_cls = fp.sum(axis=0)
        fn_cls = fn.sum(axis=0)


        macro_prec_arr = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        macro_rec_arr = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        macro_f1_arr = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))

        macro_precision = macro_prec_arr.mean()
        macro_recall = macro_rec_arr.mean()
        macro_f1 = macro_f1_arr.mean()


        tp_all = tp.sum()
        fp_all = fp.sum()
        fn_all = fn.sum()

        micro_precision = float(np.nan_to_num(tp_all / (tp_all + fp_all)))
        micro_recall = float(np.nan_to_num(tp_all / (tp_all + fn_all)))
        micro_f1 = float(np.nan_to_num(tp_all / (tp_all + 0.5 * (fp_all + fn_all))))


        scores = {

            'ce_precision': float(ce_precision),
            'ce_recall': float(ce_recall),
            'ce_f1': float(ce_f1),


            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),


            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,


            'ce_num_examples': float(len(res_bin)),
        }

        return scores
