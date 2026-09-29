import os
from pprint import pprint
from configs.hsa_her_config import parser
from dataset.hsa_her_datamodule import DataModule
from lightning_tools.hsa_her_callbacks import add_callbacks
from lightning.pytorch.callbacks import ModelCheckpoint
from models.hsa_her_stage1 import HSAHERStage1
from lightning.pytorch import seed_everything
import lightning.pytorch as pl

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)
    callbacks["callbacks"] = [
        callback for callback in callbacks["callbacks"]
        if not isinstance(callback, ModelCheckpoint)
    ]

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = HSAHERStage1.load_from_checkpoint(args.ckpt_file, args=args, strict=False)
    else:
        model = HSAHERStage1(args)

    if args.test:
        raise ValueError('Stage 1 has no report-generation test step; use validation or Stage 2 testing.')
    if args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    if args.chosen != 'vmamba' or not args.vision_model:
        raise ValueError('Stage 1 requires VMamba and an explicit --vision_model checkpoint.')
    if args.freeze_vm:
        raise ValueError('Stage 1 must train the visual encoder.')
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()
