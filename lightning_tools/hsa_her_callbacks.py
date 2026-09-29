import os
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor

def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns
