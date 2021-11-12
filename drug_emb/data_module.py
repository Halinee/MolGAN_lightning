from typing import Optional
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import sparse_molecular_dataset


class sparse_molecular_data_module(pl.LightningDataModule):
    def __init__(
        self,
        dataset_params: dict = None,
        batch_size: int = 128,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super(sparse_molecular_data_module, self).__init__()

        self.dataset = sparse_molecular_dataset(**dataset_params)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_idx, self.test_idx = train_test_split(
            range(len(self.dataset)), test_size=0.1, random_state=self.seed
        )

    def dataloader(self, split, shuffle):
        splits = {"train": self.train_idx, "test": self.test_idx}
        dataset = Subset(self.dataset, splits[split])

        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dl

    def train_dataloader(self):
        return self.dataloader(split="train", shuffle=True)

    def test_dataloader(self):
        return self.dataloader(split="test", shuffle=False)
