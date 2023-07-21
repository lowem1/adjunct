import lightning as pl
import shutil
from datasets import (
    Dataset,
    load_dataset,
    ClassLabel,
    set_caching_enabled,
    load_from_disk,
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from typing import List, Dict, Tuple, Optional, Any
from uuid import uuid1


class BaseDataModuleForLMFeaturization(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_checkpoint: str,
        tokenizer_args_map: Optional[Dict["str", Any]],
        select_columns: Optional[List["str"]],
        select_rename_map: Optional[Dict["str", "str"]],
        test_limit: Optional["int"],
        train_test_split_size: Optional["float"] = 0.2,
        train_batch_size: Optional["int"] = 16,
        val_batch_size: Optional["int"] = 16,
        test_batch_size: Optional["int"] = 16,
        data_checkpoint: Optional["str"] = "./",
    ):
        super().__init__()

        self.tokenizer, self.dataset = self.init_module(
            tokenizer_checkpoint,
            data_checkpoint,
            tokenizer_args_map,
            select_columns,
            select_rename_map,
        )
        self.data_dir = f"/tmp/{uuid1()}"
        self.train = None
        self.val = None
        self.test = None
        self.test_limit = test_limit
        self.train_test_split_size = train_test_split_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    @staticmethod
    def init(
        tokenizer_checkpoint,
        data_checkpoint,
        tokenizer_args_map,
        select_columns,
        select_rename_map,
    ) -> Tuple[Dataset, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint, **tokenizer_args_map
        )
        dataset = (
            load_dataset(data_checkpoint)["train"]
            .rename_columns(select_rename_map)
            .select_columns(select_columns)
        )
        return tokenizer, dataset

    @staticmethod
    def transform(dataset, tokenizer, **kwargs) -> None:
        print("implement me")
        return None

    def prepare_data(self):
        self.transform(self.dataset, self.tokenizer).save_to_disk(self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            full_feats = load_from_disk(self.data_dir)
            self.train, self.val = full_feats.train_test_split(
                test_size=self.train_test_split_size
            ).values()
        if stage == "test":
            full_feats = load_from_disk(self.data_dir)
            self.test = full_feats.shuffle().select(range(self.test_limit))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            sampler=RandomSampler(self.train),
            batch_size=self.train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, sampler=RandomSampler(self.val), batch_size=self.val_batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, sampler=RandomSampler(self.test), batch_size=self.test_batch_size
        )

    def teardown(self, stage):
        shutil.rmtree(self.data_dir)
