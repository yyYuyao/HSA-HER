from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.hsa_her_dataset import ParseDataset as DefaultDataset
from dataset.hsa_her_stage1_dataset import ParseDataset as ClipDataset

class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        dataset_class = ClipDataset if getattr(self.args, "data_mode", "default") == "clip" else DefaultDataset
        if stage in (None, "fit"):
            self.dataset = {"train": dataset_class(self.args, "train"),
                            "validation": dataset_class(self.args, "val")}
        elif stage == "validate":
            self.dataset = {"validation": dataset_class(self.args, "val")}
        elif stage == "test":
            self.dataset = {"test": dataset_class(self.args, "test")}
        else:
            raise ValueError(f"Unsupported data stage: {stage}")

    def train_dataloader(self):
        pass

        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def val_dataloader(self):
        pass

        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader
