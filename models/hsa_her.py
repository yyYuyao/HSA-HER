import os
import json
import time
from contextlib import nullcontext
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.metrics_clinical import CheXbertMetrics

from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType

from models.hsa_her_classification import adjust_logits, binary_classification_loss, load_class_priors
from einops import rearrange,reduce
import os
import numpy as np
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONDITIONS = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'lung opacity', 'lung lesion',
    'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax',
    'pleural effusion', 'pleural other', 'fracture', 'support devices', 'no finding',
]

from collections import OrderedDict
from models.hsa_her_private_interfaces import load_external_module

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, downsample_rate, encoder_dim, llm_dim, ffn_dim: int = 2048, **kwargs):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 2

        self.query_len = 64
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.negative_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )

        query_proj = self.norm(self.linear(query_output.last_hidden_state))

        return query_proj

class HSAHER(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.demo:
            None
        else:
            self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.proj = args.proj
        self.chosen= args.chosen
        self.llm = args.llm

        if self.chosen != 'vmamba':
            self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        else:

            from VMamba.classification.config import get_config
            from VMamba.classification.models import build_model
            from collections import OrderedDict
            class vmamba_args:
                cfg ='./VMamba/classification/configs/vssm1/vssm_base_224.yaml'
                opts= None
                batch_size=8
                zip=None
                cache_mode='part'
                pretrained = args.vision_model
                resume =None
                accumulation_steps =None
                use_checkpoint=None
                disable_amp=None
                output ='./output'
                tag =None
                throughput=None
                traincost =None
                fused_layernorm=None
                optim =None
                model_ema =True
                model_ema_decay =0.9999
                model_ema_force_cpu =False
                memory_limit_rate = -1
            config = get_config(vmamba_args)
            self.visual_encoder = build_model(config,is_pretrain=True)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        print(self.llm)
        if self.llm != 'llama2':
            self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model,)

            self.llama_tokenizer.pad_token_id = 0

            self.llama_tokenizer.bos_token_id = 0
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.bfloat16,

                )
        else:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
            self.llama_tokenizer.pad_token_id = 0
            if args.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    args.llama_model,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    args.llama_model,
                    torch_dtype=torch.bfloat16,
                )

        if args.llm_use_lora:

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False,target_modules=["q_proj", "v_proj"], r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LoRA Done')
        else:

            if self.args.llm_freeze is True:
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
            print('Loading LLAMA Done')

        self.register_buffer('base_probs', load_class_priors(args), persistent=False)

        if self.proj != 'qformer':

            self.llama_proj_moe = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.GELU(),
                nn.Linear(2048, self.llama_model.config.hidden_size)
            )
        else:
            self.llama_proj_moe = EncoderProjectorQFormer(0,encoder_dim = 1024, llm_dim = self.llama_model.config.hidden_size)

        self.projection_dim = getattr(args, 'projection_dim', 512)
        self.vision_proj = nn.Linear(self.visual_encoder.num_features, self.projection_dim)
        self.cls_head = nn.Linear(self.projection_dim, 14)
        nn.init.normal_(self.vision_proj.weight, std=0.001)
        if self.vision_proj.bias is not None:
            nn.init.constant_(self.vision_proj.bias, 0)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        expert_dim = 1024
        self.expert_dim = expert_dim
        self.her_router = load_external_module(getattr(args, "her_factory", None))
        self.latest_debug_shapes = {}
        self.enable_nan_diagnostics = getattr(args, 'enable_nan_diagnostics', True)


        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = self.args.instruction
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = float("-inf")

        self.context_base = []
        self.negative_samples, self.positive_samples = None, None

        chexbert_path = getattr(args, 'chexbert_path', None)
        bert_model_path = getattr(args, 'bert_model_path', 'bert-base-uncased')
        self.chexbert_path = chexbert_path
        self.bert_model_path = bert_model_path
        self.chexbert_batch_size = args.batch_size
        self.chexbert_metrics = None

        if getattr(args, 'clip_stage_ckpt', None) is not None:
            self.load_stage1_checkpoint(args.clip_stage_ckpt)

        if args.delta_file is not None:
            checkpoint = torch.load(args.delta_file, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict=checkpoint['model'], strict=False)
            self.val_score = checkpoint.get('validation_score', float('-inf'))
            print(f'Load checkpoint from {args.delta_file}')

    def setup(self, stage=None):
        pass

        if self.trainer.is_global_zero:
            if self.chexbert_path is None:
                raise ValueError("--chexbert_path is required for report evaluation.")
            if getattr(self, 'chexbert_metrics', None) is None:
                self.chexbert_metrics = CheXbertMetrics(
                    self.chexbert_path,
                    self.chexbert_batch_size,
                    self.device,
                    self.bert_model_path
                )
                print(f'✅ CheXbert metrics initialized ONLY on main device: {self.device} to save VRAM.')

    def score(self, ref, hypo):
        pass

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        if self.args.dataset == 'chinese':
            hypo = {k: [' '.join(vi) for vi in v] for k, v in hypo.items()}
            ref = {k: [' '.join(vi) for vi in v] for k, v in ref.items()}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        desired_order = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]
        return {key: final_scores[key] for key in desired_order if key in final_scores}

    def _format_sample_debug_context(self, samples):
        if samples is None:
            return "sample_context=unavailable"

        parts = []
        for key in ("id", "region_txt_path", "retrieval_txt_sources"):
            value = samples.get(key)
            if value is None:
                continue
            parts.append(f"{key}={value}")

        if self.latest_debug_shapes:
            parts.append(f"latest_debug_shapes={self.latest_debug_shapes}")
        return ", ".join(parts) if parts else "sample_context=empty"

    def _assert_finite_tensor(self, tensor_name, tensor, samples=None):
        if (not self.enable_nan_diagnostics) or tensor is None:
            return

        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return

        finite_values = tensor[finite_mask]
        max_abs = float(finite_values.abs().max().item()) if finite_values.numel() > 0 else None
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "posinf_count": int(torch.isposinf(tensor).sum().item()),
            "neginf_count": int(torch.isneginf(tensor).sum().item()),
            "max_abs_finite": max_abs,
        }
        raise RuntimeError(
            f"Non-finite tensor detected at {tensor_name}: stats={stats}, "
            f"{self._format_sample_debug_context(samples)}"
        )

    def encode_img(self, images,global_only=False,global_only_return=False,use_feature_mean=True,featuremap_folder=None):
        image_embeds = []
        for image in images:
            device = image.device
            if self.chosen == 'vmamba':
                if global_only:
                    image_embed = self.visual_encoder(image,global_only)
                else:
                    image_embed = self.visual_encoder(image,global_only,featuremap_folder=featuremap_folder)
                    image_embed = rearrange(image_embed, 'b h w e -> b (h w) e')
            elif self.chosen == 'vim':
                image_embed = self.visual_encoder(image,return_features=True)
            else:
                if global_only:
                    image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
                    image_embed = reduce(image_embed, 'b l e -> b e','mean')
                else:
                    image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
        if self.args.use_feature_mean or global_only is True:
            image_embeds = torch.stack(image_embeds).mean(0)
            if global_only_return:
                return image_embeds,None
        else:
            if len(image_embeds)==1 :

                image_embeds = image_embeds + image_embeds
            image_embeds = torch.concat(image_embeds,dim=1)

        atts_img = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        return image_embeds, atts_img

    def prompt_wrap(self, img_embeds):
        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_before_embeds = self.token_embed(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.token_embed(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts = torch.ones(wrapped_embeds.size()[:-1], dtype=torch.long, device=img_embeds.device)
        return wrapped_embeds, wrapped_atts

    def build_label_prompt_strings(self, cls_preds):
        probabilities = torch.sigmoid(cls_preds)
        states = (probabilities >= 0.5).int().cpu().tolist()
        prompts = []
        for sample_states in states:
            label_pairs = [
                f'{condition}: {"positive" if state else "negative"}'
                for condition, state in zip(CONDITIONS, sample_states)
            ]
            prompts.append(f"CheXpert labels: {'; '.join(label_pairs)}. Report:")
        return prompts, probabilities

    def build_label_prompt_tokens(self, cls_preds, device):
        cls_prompt_strs, cls_preds_softmax = self.build_label_prompt_strings(cls_preds)
        cls_tokens = self.llama_tokenizer(
            cls_prompt_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        cls_embeds = self.token_embed(cls_tokens.input_ids)
        return cls_prompt_strs, cls_preds_softmax, cls_tokens, cls_embeds

    def project_moe_tokens(self, moe_tokens):
        if self.proj != 'qformer':
            projected_tokens = self.llama_proj_moe(moe_tokens)
        else:
            token_attention_mask = torch.ones(
                moe_tokens.size()[:-1], dtype=torch.long, device=moe_tokens.device
            )
            projected_tokens = self.llama_proj_moe.forward(moe_tokens, token_attention_mask)
        return self.layer_norm(projected_tokens)

    def apply_logit_adjustment(self, cls_preds, device):
        return adjust_logits(cls_preds, self.base_probs)

    def compute_cls_logits(self, avg_embeds):
        cls_features = self.vision_proj(avg_embeds)
        return self.cls_head(cls_features)

    def load_stage1_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        current_state_dict = self.state_dict()

        remapped_state_dict = {}
        loaded_items = []
        skipped_items = []

        if 'vision_proj.weight' in state_dict and state_dict['vision_proj.weight'].shape == current_state_dict['vision_proj.weight'].shape:
            remapped_state_dict['vision_proj.weight'] = state_dict['vision_proj.weight']
            loaded_items.append('vision_proj.weight')
        elif 'vision_proj.weight' in state_dict:
            skipped_items.append('vision_proj.weight')
        if 'vision_proj.bias' in state_dict and state_dict['vision_proj.bias'].shape == current_state_dict['vision_proj.bias'].shape:
            remapped_state_dict['vision_proj.bias'] = state_dict['vision_proj.bias']
            loaded_items.append('vision_proj.bias')
        elif 'vision_proj.bias' in state_dict:
            skipped_items.append('vision_proj.bias')

        if 'image_classifier.weight' in state_dict and state_dict['image_classifier.weight'].shape == current_state_dict['cls_head.weight'].shape:
            remapped_state_dict['cls_head.weight'] = state_dict['image_classifier.weight']
            loaded_items.append('image_classifier.weight->cls_head.weight')
        elif 'image_classifier.weight' in state_dict:
            skipped_items.append('image_classifier.weight->cls_head.weight')
        if 'image_classifier.bias' in state_dict and state_dict['image_classifier.bias'].shape == current_state_dict['cls_head.bias'].shape:
            remapped_state_dict['cls_head.bias'] = state_dict['image_classifier.bias']
            loaded_items.append('image_classifier.bias->cls_head.bias')
        elif 'image_classifier.bias' in state_dict:
            skipped_items.append('image_classifier.bias->cls_head.bias')

        for key, value in state_dict.items():
            if key.startswith('visual_encoder.') and key in current_state_dict and value.shape == current_state_dict[key].shape:
                remapped_state_dict[key] = value
            elif key.startswith('visual_encoder.') and key in current_state_dict:
                skipped_items.append(key)

        required = {
            'vision_proj.weight': 'vision_proj.weight',
            'vision_proj.bias': 'vision_proj.bias',
            'image_classifier.weight': 'cls_head.weight',
            'image_classifier.bias': 'cls_head.bias',
        }
        for source_name, target_name in required.items():
            if source_name not in state_dict or target_name not in remapped_state_dict:
                raise ValueError(
                    f'Stage 1 checkpoint is missing or has incompatible {source_name}; '
                    'a checkpoint trained with the binary HSA-HER classifier is required.'
                )
        if not any(key.startswith('visual_encoder.') for key in remapped_state_dict):
            raise ValueError('Stage 1 checkpoint contains no compatible VMamba weights.')

        msg = self.load_state_dict(remapped_state_dict, strict=False)
        print(f'Load stage-1 checkpoint from {ckpt_path}')
        print(f'Stage-1 initialized params: {loaded_items}')
        if skipped_items:
            preview = skipped_items[:10]
            print(f'Stage-1 skipped params due to shape mismatch ({len(skipped_items)}): {preview}')
        if msg.missing_keys:
            print(f'Stage-1 init missing keys: {msg.missing_keys}')
        if msg.unexpected_keys:
            print(f'Stage-1 init unexpected keys: {msg.unexpected_keys}')

    def prepare_text_features(self, samples, device, target_dtype):
        retrieval_txt = samples['retrieval_txt'].to(device=device, dtype=target_dtype)
        region_txt = samples['region_txt'].to(device=device, dtype=target_dtype)

        if region_txt.dim() == 4:
            region_txt = region_txt.view(region_txt.size(0), -1, region_txt.size(-1))
        if retrieval_txt.dim() == 3:
            retrieval_txt = retrieval_txt.unsqueeze(1)
        if retrieval_txt.dim() == 4:
            retrieval_txt = retrieval_txt.view(
                retrieval_txt.size(0), retrieval_txt.size(1) * retrieval_txt.size(2), retrieval_txt.size(3)
            )

        self._assert_finite_tensor("inputs.region_txt", region_txt, samples)
        self._assert_finite_tensor("inputs.retrieval_txt", retrieval_txt, samples)
        return region_txt, retrieval_txt

    def encode_fused_img_tokens(self, samples, image, img_embeds):
        region_txt, retrieval_txt = self.prepare_text_features(samples, image[0].device, img_embeds.dtype)
        fused_img_tokens = self.her_router(img_embeds, region_txt, retrieval_txt)
        if fused_img_tokens.ndim != 3 or fused_img_tokens.shape[:2] != img_embeds.shape[:2]:
            raise ValueError("External HER module must return [batch, visual_tokens, expert_dim].")
        fused_img_embeds = self.project_moe_tokens(fused_img_tokens)
        prompt_embeds, prompt_atts = self.prompt_wrap(fused_img_embeds)
        return {"prompt_embeds": prompt_embeds, "prompt_atts": prompt_atts}

    def build_lm_inputs(self, prompt_embeds, prompt_atts, cls_tokens, cls_embeds, report_tokens=None):
        batch_size = prompt_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=prompt_embeds.device
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.token_embed(bos)
        atts_bos = torch.ones((batch_size, 1), dtype=torch.long, device=prompt_embeds.device)

        input_parts = [bos_embeds, cls_embeds, prompt_embeds]
        mask_parts = [atts_bos, cls_tokens.attention_mask, prompt_atts]

        targets = None
        if report_tokens is not None:
            report_embeds = self.token_embed(report_tokens.input_ids)
            input_parts.append(report_embeds)
            mask_parts.append(report_tokens.attention_mask)

            targets = report_tokens.input_ids.masked_fill(report_tokens.input_ids == 0, -100)
            empty_targets = torch.full(
                (prompt_embeds.shape[0], 1 + cls_embeds.shape[1] + prompt_embeds.shape[1]),
                -100,
                dtype=torch.long,
                device=prompt_embeds.device,
            )
            targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = torch.cat(input_parts, dim=1)
        attention_mask = torch.cat(mask_parts, dim=1)
        assert inputs_embeds.shape[:2] == attention_mask.shape
        if targets is not None:
            assert inputs_embeds.size(1) == targets.size(1)
        return inputs_embeds, attention_mask, targets

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        self._assert_finite_tensor("forward.img_embeds", img_embeds, samples)
        cls_labels = samples['cls_labels'][:, :14].to(image[0].device)

        avg_embeds = img_embeds.mean(dim=1)

        cls_preds = self.compute_cls_logits(avg_embeds)
        self._assert_finite_tensor("forward.cls_preds_before_adjust", cls_preds, samples)

        cls_preds = self.apply_logit_adjustment(cls_preds, img_embeds.device)
        self._assert_finite_tensor("forward.cls_preds_after_adjust", cls_preds, samples)

        loss_cls = binary_classification_loss(cls_preds, cls_labels)

        fusion_outputs = self.encode_fused_img_tokens(samples, image, img_embeds)
        prompt_embeds = fusion_outputs["prompt_embeds"]
        prompt_atts = fusion_outputs["prompt_atts"]

        _, _, cls_tokens, cls_embeds = self.build_label_prompt_tokens(cls_preds, image[0].device)
        self._assert_finite_tensor("forward.cls_embeds", cls_embeds, samples)

        self.llama_tokenizer.padding_side = "right"
        report_tokens = self.llama_tokenizer(
            [t + self.end_sym for t in samples["input_text"]],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        inputs_embeds, attention_mask, targets = self.build_lm_inputs(
            prompt_embeds,
            prompt_atts,
            cls_tokens,
            cls_embeds,
            report_tokens=report_tokens,
        )
        self._assert_finite_tensor("forward.inputs_embeds", inputs_embeds, samples)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        valid_report_tokens = targets[:, 1:].ne(-100).sum()
        if valid_report_tokens.item() == 0:
            raise ValueError("Language-model targets contain no report tokens.")
        loss_lm = outputs.loss * valid_report_tokens.to(outputs.loss.dtype) / targets.size(0)
        self._assert_finite_tensor("forward.loss_lm", loss_lm, samples)

        cls_w = getattr(self.args, 'cls_weight', 4.0)

        loss = loss_lm + (cls_w * loss_cls)
        self._assert_finite_tensor("forward.loss", loss, samples)

        return {
            "loss": loss,
            "loss_lm": loss_lm,
            "loss_cls": loss_cls,
        }

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        save_obj['validation_score'] = eval_res['val_score']
        torch.save(save_obj, save_to)
        with open(os.path.join(self.hparams.savedmodel_path, 'best_checkpoint.json'), 'w', encoding='utf-8') as stream:
            json.dump({'checkpoint': os.path.basename(save_to),
                       'validation_score': eval_res['val_score'],
                       'epoch': current_epoch}, stream, ensure_ascii=False)

    def validation_step(self, samples, batch_idx, dataloader_idx=0):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",

            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        avg_embeds = img_embeds.mean(dim=1)

        cls_preds = self.compute_cls_logits(avg_embeds)
        cls_preds = self.apply_logit_adjustment(cls_preds, image[0].device)

        _, cls_preds_softmax, cls_tokens, cls_embeds = self.build_label_prompt_tokens(
            cls_preds, image[0].device
        )

        raw_pos_probs = cls_preds_softmax.detach().float().cpu().numpy()
        raw_cls_labels = samples['cls_labels'][:, :14].float().cpu().numpy()

        fusion_outputs = self.encode_fused_img_tokens(samples, image, img_embeds)
        prompt_embeds = fusion_outputs["prompt_embeds"]
        prompt_atts = fusion_outputs["prompt_atts"]
        inputs_embeds, attention_mask, _ = self.build_lm_inputs(
            prompt_embeds, prompt_atts, cls_tokens, cls_embeds
        )

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]

        if dataloader_idx == 0:
            self.val_step_outputs.append({
                "hypo": hypo,
                "ref": ref,
                "id": samples["id"],
                "raw_pos_probs": raw_pos_probs,
                "cls_labels": raw_cls_labels
            })
        else:
            self.test_step_outputs.append({
                "hypo": hypo,
                "ref": ref,
                "id": samples["id"]
            })

        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace('!', '')
        return output_text

    def token_embed(self, input_ids):
        return self.llama_model.get_input_embeddings()(input_ids)

    def format_metric_value(self, value):
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.6f}"
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=4, separator=', ')
        return str(value)

    def format_metric_lines(self, metrics):
        if not metrics:
            return ["(empty)"]

        key_width = max(len(str(key)) for key in metrics.keys())
        return [
            f"{str(key):<{key_width}} : {self.format_metric_value(value)}"
            for key, value in metrics.items()
        ]

    def build_metric_block(self, title, sections):
        lines = ["", "=" * 72, title, "=" * 72]
        for section_name, metrics in sections:
            lines.append(f"[{section_name}]")
            lines.extend(self.format_metric_lines(metrics))
            lines.append("")
        return "\n".join(lines).rstrip()

    def on_validation_epoch_end(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_outputs = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_outputs, self.val_step_outputs)
            all_outputs = [item for sublist in gathered_outputs for item in sublist]
        else:
            all_outputs = self.val_step_outputs

        validation_metrics = None
        validation_ce_metrics = None
        validation_epoch = self.current_epoch

        if self.trainer.is_global_zero:
            unique_results = {}

            for batch_out in all_outputs:

                for r, h, idx, pos_p, lbl in zip(batch_out['ref'], batch_out['hypo'], batch_out['id'],
                                                 batch_out['raw_pos_probs'], batch_out['cls_labels']):
                    if idx not in unique_results:
                        unique_results[idx] = {
                            'ref': [r],
                            'hypo': [h],
                            'raw_pos_probs': pos_p,
                            'cls_labels': lbl
                        }

            ref = {k: v['ref'] for k, v in unique_results.items()}
            hypo = {k: v['hypo'] for k, v in unique_results.items()}

            eval_res = self.score(ref=ref, hypo=hypo)
            ref_flat = [item for sublist in ref.values() for item in sublist]
            hypo_flat = [item for sublist in hypo.values() for item in sublist]
            eval_ce = self.chexbert_metrics.compute(ref_flat, hypo_flat)

            self.log_dict(eval_res, sync_dist=False, rank_zero_only=True, logger=True)
            self.log_dict(eval_ce, sync_dist=False, rank_zero_only=True, logger=True)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step

            json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w',
                                 encoding='utf-8'), ensure_ascii=False)
            json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'), ensure_ascii=False)

            validation_metrics = eval_res
            validation_ce_metrics = eval_ce
            validation_epoch = current_epoch

            val_score = 0.5 * eval_res["Bleu_4"] + 0.5 * eval_res["CIDEr"]
            if not np.isfinite(val_score):
                raise ValueError("Validation selection score must be finite.")

            eval_res["val_score"] = val_score

            self._is_best_epoch = False

            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
                self._is_best_epoch = True

        if self.trainer.is_global_zero and validation_metrics is not None and validation_ce_metrics is not None:
            self.print(self.build_metric_block(
                f"Epoch {validation_epoch} Validation Summary",
                [
                    ("NLP Metrics", validation_metrics),
                    ("CheXbert Metrics", validation_ce_metrics),
                ]
            ))

        self.val_step_outputs.clear()
        self.test_step_outputs.clear()

    def demo_test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        avg_embeds = img_embeds.mean(dim=1)

        cls_preds = self.compute_cls_logits(avg_embeds)
        cls_preds = self.apply_logit_adjustment(cls_preds, image[0].device)

        _, _, cls_tokens, cls_embeds = self.build_label_prompt_tokens(cls_preds, image[0].device)

        fusion_outputs = self.encode_fused_img_tokens(samples, image, img_embeds)
        prompt_embeds = fusion_outputs["prompt_embeds"]
        prompt_atts = fusion_outputs["prompt_atts"]
        inputs_embeds, attention_mask, _ = self.build_lm_inputs(
            prompt_embeds, prompt_atts, cls_tokens, cls_embeds
        )

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.args.beam_size,
            do_sample=self.args.do_sample,
            min_new_tokens=self.args.min_new_tokens,
            max_new_tokens=self.args.max_new_tokens,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            temperature=self.args.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        image = samples["image"]

        featuremap_folder=None
        if self.args.featuremap:
            featuremap_folder = './featuremap/'+str(samples["id"][0])
            os.system(f'mkdir -p {featuremap_folder}')
        img_embeds, atts_img = self.encode_img(image,featuremap_folder=featuremap_folder)

        avg_embeds = img_embeds.mean(dim=1)

        cls_preds = self.compute_cls_logits(avg_embeds)
        cls_preds = self.apply_logit_adjustment(cls_preds, image[0].device)

        _, _, cls_tokens, cls_embeds = self.build_label_prompt_tokens(cls_preds, image[0].device)

        fusion_outputs = self.encode_fused_img_tokens(samples, image, img_embeds)
        prompt_embeds = fusion_outputs["prompt_embeds"]
        prompt_atts = fusion_outputs["prompt_atts"]
        inputs_embeds, attention_mask, _ = self.build_lm_inputs(
            prompt_embeds, prompt_atts, cls_tokens, cls_embeds
        )

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        if self.args.featuremap:
            file_path = f"{featuremap_folder}/predict.txt"
            txt = 'hypo: '.join(hypo)
            txt = txt+ '\n'+'ref: '.join(ref)+ '\n'
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(txt)
        self.test_step_outputs.append({
            "hypo": hypo,
            "ref": ref,
            "id": samples["id"],
        })
        return hypo, ref

    def on_test_epoch_end(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_outputs = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_outputs, self.test_step_outputs)
            all_outputs = [item for shard in gathered_outputs for item in shard]
        else:
            all_outputs = self.test_step_outputs
        if self.trainer.is_global_zero:
            unique_results = {}
            for batch_out in all_outputs:
                for reference, prediction, identifier in zip(batch_out['ref'], batch_out['hypo'], batch_out['id']):
                    unique_results.setdefault(identifier, {'ref': [reference], 'hypo': [prediction]})
            ref = {key: value['ref'] for key, value in unique_results.items()}
            hypo = {key: value['hypo'] for key, value in unique_results.items()}
            eval_res = self.score(ref=ref, hypo=hypo)
            ref_flat = [item for values in ref.values() for item in values]
            hypo_flat = [item for values in hypo.values() for item in values]
            eval_ce = self.chexbert_metrics.compute(ref_flat, hypo_flat)
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            with open(os.path.join(result_folder, 'test_result.json'), 'w', encoding='utf-8') as stream:
                json.dump(hypo, stream, ensure_ascii=False)
            with open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8') as stream:
                json.dump(ref, stream, ensure_ascii=False)
            self.print(self.build_metric_block('Test Summary', [('NLP Metrics', eval_res), ('CheXbert Metrics', eval_ce)]))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):

        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
