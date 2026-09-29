import os
import json
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from transformers import SwinModel
from torch.nn import functional as F
from models.hsa_her_private_interfaces import load_external_module
from peft import get_peft_model, LoraConfig, TaskType

from models.hsa_her_classification import (
    adjust_logits, binary_classification_accuracy, binary_classification_loss, load_class_priors
)
from einops import rearrange, reduce

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BIOCLINICALBERT_REPO_ID = "emilyalsentzer/Bio_ClinicalBERT"

def _is_hf_model_dir(path):
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

def resolve_bioclinicalbert_source(preferred_path=None):
    candidate_roots = []
    for candidate in (
        preferred_path,
        os.environ.get("BIOCLINICALBERT_PATH"),
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--emilyalsentzer--Bio_ClinicalBERT"),
    ):
        if candidate:
            candidate_roots.append(os.path.expanduser(candidate))

    seen = set()
    for candidate_root in candidate_roots:
        if candidate_root in seen:
            continue
        seen.add(candidate_root)

        if _is_hf_model_dir(candidate_root):
            return candidate_root

        snapshots_dir = os.path.join(candidate_root, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        snapshot_candidates = [
            os.path.join(snapshots_dir, snapshot_name)
            for snapshot_name in os.listdir(snapshots_dir)
            if _is_hf_model_dir(os.path.join(snapshots_dir, snapshot_name))
        ]
        if snapshot_candidates:
            snapshot_candidates.sort(key=os.path.getmtime, reverse=True)
            return snapshot_candidates[0]

    return BIOCLINICALBERT_REPO_ID

def compute_itc(image_features, text_features, logit_scale):
    pass

    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64).to(image_features.device)

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2
    return loss

class HSAHERStage1(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(args, 'demo', False):
            pass
        else:
            self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.proj = getattr(args, 'proj', None)
        self.chosen = getattr(args, 'chosen', 'vmamba')

        if self.chosen != 'vmamba':

            try:
                self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
            except Exception as e:
                print(f"=> Warning: pretrained vision weights not found for {args.vision_model}; initializing Swin from config only.")
                config = AutoConfig.from_pretrained(args.vision_model)
                self.visual_encoder = SwinModel(config)
        else:
            from VMamba.classification.config import get_config
            from VMamba.classification.models import build_model
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
            self.visual_encoder = build_model(config, is_pretrain=True)

            if os.path.exists(config.MODEL.PRETRAINED):
                checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
                if 'model' in checkpoint:
                    self.visual_encoder.load_state_dict(checkpoint['model'], strict=False)
                    print(f"=> loaded pretrained vssm model successfully from {config.MODEL.PRETRAINED}")
            else:
                print(f"=> Warning: pretrained VMamba weights not found at {config.MODEL.PRETRAINED}.")
                print("=> Skipping base-weight loading and keeping the initialized VMamba backbone.")

        if getattr(args, 'vis_use_lora', False):
            if self.chosen == 'vmamba':
                target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]
            else:
                target_modules = ["query", "value"]
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=target_modules,
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif getattr(args, 'freeze_vm', False):
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print(f"Loading text encoder : {getattr(args, 'text_encoder_type', 'Bio_ClinicalBERT')}...")
        self.text_encoder_type = getattr(args, 'text_encoder_type', 'Bio_ClinicalBERT')
        if self.text_encoder_type == 'Bio_ClinicalBERT':
            text_encoder_source = resolve_bioclinicalbert_source(
                getattr(args, "text_encoder_path", None)
            )
            print(f"Using Bio_ClinicalBERT source: {text_encoder_source}")
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_source)
            self.text_encoder = AutoModel.from_pretrained(text_encoder_source)
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = self.tokenizer.cls_token_id

        self.projection_dim = getattr(args, 'projection_dim', 512)
        self.use_itc = getattr(args, 'use_itc', True)
        self.use_cls = getattr(args, 'use_cls', True)
        self.use_homo = getattr(args, 'use_homo', True)
        self.alignment_loss = load_external_module(getattr(args, 'alignment_factory', None)) if self.use_homo else None
        self.itc_weight = getattr(args, 'itc_weight', 1.0)
        self.cls_weight = getattr(args, 'clip_cls_weight', getattr(args, 'id_loss_weight', 1.0))
        self.image_cls_weight = getattr(args, 'image_cls_weight', 1.0)
        self.homo_weight = getattr(args, 'homo_weight', 1.0)

        vision_dim = self.visual_encoder.num_features if hasattr(self.visual_encoder, 'num_features') else getattr(args, 'vision_dim', 1024)
        self.vision_proj = nn.Linear(vision_dim, self.projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim)

        self.temperature = getattr(args, 'temperature', 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / self.temperature).log())

        self.num_labels = 14
        self.image_classifier = nn.Linear(self.projection_dim, self.num_labels)
        self.register_buffer('base_probs', load_class_priors(args), persistent=False)

        for param in self.parameters():
            if param.data is not None:
                param.data = param.data.contiguous()

        print("Successfully ensured all parameters are contiguous for DeepSpeed.")
        nn.init.normal_(self.image_classifier.weight, std=0.001)
        nn.init.constant_(self.image_classifier.bias, 0)

        if getattr(args, 'delta_file', None) is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def encode_img(self, images, global_only=False, featuremap_folder=None):
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = [images]

        global_embeds = []
        grid_embeds = []

        for image in images:
            if self.chosen == 'vmamba':

                base_model = self.visual_encoder
                if hasattr(base_model, 'base_model'):
                    base_model = base_model.base_model.model
                elif hasattr(base_model, 'module'):
                    base_model = base_model.module

                if hasattr(base_model, 'forward_features'):
                    feat = base_model.forward_features(image)
                else:
                    feat = base_model(image)

                if len(feat.shape) == 2:
                    x = image
                    if hasattr(base_model, 'patch_embed'):
                        x = base_model.patch_embed(x)
                    if hasattr(base_model, 'pos_drop'):
                        x = base_model.pos_drop(x)
                    if hasattr(base_model, 'layers'):
                        for layer in base_model.layers:
                            x = layer(x)
                    if hasattr(base_model, 'norm'):
                        x = base_model.norm(x)
                    feat = x

                if len(feat.shape) == 4:

                    if feat.shape[-1] > feat.shape[1]:

                        feat = rearrange(feat, 'b h w c -> b (h w) c')
                    else:

                        feat = rearrange(feat, 'b c h w -> b (h w) c')

                grid_embeds.append(feat)
                global_embeds.append(feat.mean(dim=1))

            elif self.chosen == 'vim':
                feat = self.visual_encoder(image, return_features=True)
                grid_embeds.append(feat)
                global_embeds.append(feat.mean(dim=1))
            else:
                feat = self.visual_encoder(image)['last_hidden_state']
                grid_embeds.append(feat)
                global_embeds.append(feat.mean(dim=1))

        image_global = torch.stack(global_embeds).mean(0)
        image_grid = torch.stack(grid_embeds).mean(0)

        return image_global, image_grid

    def encode_txt(self, text_tokens):
        if self.text_encoder_type == 'Bio_ClinicalBERT':
            text_features = self.text_encoder(text_tokens['input_ids'], attention_mask=text_tokens['attention_mask'])["last_hidden_state"]

        global_features = text_features[:, 0, :]
        global_features = self.text_proj(global_features)

        seq_features = self.text_proj(text_features)
        return global_features, seq_features

    def tokenize_reports(self, report, device):
        return self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=150
        ).to(device)

    def forward(self, samples):
        image = samples["image"]

        report = samples["input_text"]
        text_tokens = self.tokenize_reports(report, image[0].device)

        image_features, _ = self.encode_img(image)

        image_features = self.vision_proj(image_features)

        text_features, _ = self.encode_txt(text_tokens)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        dummy_loss = 0.0 * (image_features.sum() + text_features.sum())

        loss_itc = dummy_loss
        if self.use_itc:
            loss_itc = compute_itc(image_features, text_features, logit_scale)

        labels = samples.get('labels', None)
        loss_cls = dummy_loss
        loss_cls_img = dummy_loss
        loss_homo = dummy_loss
        cls_acc = torch.tensor(0.0, device=image_features.device)

        if self.use_cls and labels is None:
            raise ValueError('Stage 1 classification requires disease labels.')
        if labels is not None:

            if self.use_cls:
                logits_img = adjust_logits(self.image_classifier(image_features), self.base_probs)
                loss_cls_img = binary_classification_loss(logits_img, labels)
                loss_cls = self.image_cls_weight * loss_cls_img
                cls_acc = binary_classification_accuracy(logits_img, labels)

        if self.use_homo:
            loss_homo = self.alignment_loss(image_features, text_features)

        total_loss = (
            (self.itc_weight * loss_itc) +
            (self.cls_weight * loss_cls) +
            (self.homo_weight * loss_homo)
        )

        return {
            "loss": total_loss,
            "loss_itc": loss_itc,
            "loss_cls": loss_cls,
            "loss_homo": loss_homo,
            "cls_acc": cls_acc,
        }

    def training_step(self, batch, batch_idx):
        result = self(batch)
        if result is not None:
            log_dict = {f"train_{k}": v for k, v in result.items() if "features" not in k}
            self.log_dict(log_dict, prog_bar=True)
        return result["loss"]

    def save_checkpoint(self):
        if not getattr(self.trainer, 'is_global_zero', True):
            return

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
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path,
            'checkpoints',
            "checkpoint_epoch{}_step{}.pth".format(current_epoch, global_step),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        result = self(samples)
        self.log("val_loss", result["loss"], sync_dist=True, prog_bar=True)
        self.log("val_loss_itc", result["loss_itc"], sync_dist=True)
        self.log("val_loss_cls", result["loss_cls"], sync_dist=True)
        self.log("val_loss_homo", result["loss_homo"], sync_dist=True)
        self.log("val_cls_acc", result["cls_acc"], sync_dist=True)
        return result["loss"]

    def on_validation_epoch_end(self):
        self.save_checkpoint()

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
