import os
import time
from pprint import pprint
from configs.hsa_her_config import parser
from dataset.hsa_her_datamodule import DataModule
from lightning_tools.hsa_her_callbacks import add_callbacks
from models.hsa_her import HSAHER
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class DebugLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.val_start_time = 0
        self.test_start_time = 0

        self.train_epoch_start_time = 0
        self.last_log_time = 0
        self.last_log_batch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_batch = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        total_batches = trainer.num_training_batches

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            current_time = time.time()

            elapsed_seconds = current_time - self.train_epoch_start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

            steps_passed = (batch_idx + 1) - self.last_log_batch
            time_passed = current_time - self.last_log_time
            speed = steps_passed / time_passed if time_passed > 0 else 0

            remaining_batches = total_batches - (batch_idx + 1)
            eta_seconds = remaining_batches / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            metrics = trainer.progress_bar_metrics
            metric_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])

            print(f"Epoch {trainer.current_epoch}: [Train] Batch {batch_idx+1}/{total_batches} [{elapsed_str}<{eta_str}, {speed:.2f} it/s] | {metric_str}")

            self.last_log_time = current_time
            self.last_log_batch = batch_idx + 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start_time = time.time()

        self.val_epoch_start_time = time.time()
        self.val_last_log_time = time.time()
        self.val_last_log_batch = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        total_batches = trainer.num_val_batches[0] if trainer.num_val_batches else "?"

        if total_batches != "?" and ((batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches):
            current_time = time.time()

            elapsed_seconds = current_time - self.val_epoch_start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

            steps_passed = (batch_idx + 1) - self.val_last_log_batch
            time_passed = current_time - self.val_last_log_time
            speed = steps_passed / time_passed if time_passed > 0 else 0

            remaining_batches = total_batches - (batch_idx + 1)
            eta_seconds = remaining_batches / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            print(f"Epoch {trainer.current_epoch}: [Val] Batch {batch_idx+1}/{total_batches} [{elapsed_str}<{eta_str}, {speed:.2f} it/s] 生成完毕")

            self.val_last_log_time = current_time
            self.val_last_log_batch = batch_idx + 1

    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.val_start_time
        print(f"\n==========> [Timer] Epoch {trainer.current_epoch} 验证阶段 (Validation) 耗时: {elapsed:.2f} 秒 <==========\n")

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_start_time = time.time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        total_batches = trainer.num_test_batches[0] if trainer.num_test_batches else "?"
        print(f"Epoch {trainer.current_epoch}: [Test] Batch {batch_idx+1}/{total_batches} 生成完毕")

    def on_test_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.test_start_time
        print(f"\n==========> [Timer] Epoch {trainer.current_epoch} 测试阶段 (Test) 耗时: {elapsed:.2f} 秒 <==========\n")

def train(args):
    import torch
    if args.dev_form == "test":
        raise ValueError("Validation must use the val split; use --test for independent testing.")
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    my_callbacks = callbacks["callbacks"]
    my_callbacks.append(DebugLoggerCallback())

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,

        limit_val_batches=1.0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,

        max_epochs = args.max_epochs,
        val_check_interval = args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=False,
        callbacks=my_callbacks,
        logger=callbacks["loggers"]
    )

    model = HSAHER(args)
    if args.ckpt_file is not None:
        state = torch.load(args.ckpt_file, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=False)
        model.val_score = state.get('validation_score', float('-inf'))

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():

    args = parser.parse_args()
    if args.chosen != 'vmamba' or not args.vision_model:
        raise ValueError('Stage 2 requires VMamba and an explicit --vision_model checkpoint.')
    expected_llm = 'qwen' if args.dataset == 'iu_xray' else 'llama2'
    if args.dataset not in {'chexpert_plus', 'iu_xray', 'mimic_cxr'} or args.llm != expected_llm:
        raise ValueError('IU X-Ray requires Qwen; CheXpert Plus and MIMIC-CXR require Llama-2.')
    if not args.llm_freeze or args.llm_use_lora or args.freeze_vm or args.vis_use_lora:
        raise ValueError('The paper protocol freezes the LLM and fully fine-tunes VMamba.')
    if not (args.test or args.validate) and not args.clip_stage_ckpt:
        raise ValueError('Stage 2 training requires --clip_stage_ckpt from Stage 1.')
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()
