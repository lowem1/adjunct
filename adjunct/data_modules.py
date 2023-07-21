import os
import lightning as pl
from transformers import AutoTokenizer
from datasets import (
    Dataset,
    load_dataset,
    ClassLabel,
    set_caching_enabled,
    load_from_disk,
    DatasetDict,
    concatenate_datasets
)

from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Tuple
from uuid import uuid1
import abc
import nlpaug.augmenter.char as nac
import shutil

set_caching_enabled(False)


# TODO: Refactor for general sequence to label classification task
class ICD10DataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer_checkpoint: str, data_checkpoint: str = "./", limit: int = None
    ):
        super().__init__()

        self.tokenizer, self.dataset = self.init_module(
            tokenizer_checkpoint, data_checkpoint
        )
        self.labeler = ClassLabel(
            names=list(self.dataset.to_pandas()["labels"].unique())
        )
        self.n_labels = len(self.labeler.names)
        self.data_dir = f"/tmp/{uuid1()}"
        self.train = None
        self.val = None
        self.test = None

    @staticmethod
    def init_module(
        tokenizer_checkpoint, data_checkpoint
    ) -> Tuple[Dataset, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        dataset = (
            load_dataset(data_checkpoint)["train"]
            .rename_columns({"group_no": "labels", "text": "line_data"})
            .select_columns(["labels", "line_data"])
        )
        return tokenizer, dataset
    
    @staticmethod
    def transform(dataset, tokenizer) -> DatasetDict:
        tokenizer_fn = lambda _, col: tokenizer.batch_encode_plus(
            _[col],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        labeler = ClassLabel(
            names=list(dataset.to_pandas()["labels"].unique())
         )
        
        labeling_fn = lambda _, labeler: dict(
            labels=labeler.str2int(_["labels"])
        )
        
        transformed_dataset = (
            dataset.map(
                tokenizer_fn, fn_kwargs=(dict(col="line_data")), batched=True,
            ).select_columns(["input_ids", "attention_mask", "labels"])
            .map(labeling_fn, fn_kwargs=dict(labeler=labeler))
            .with_format(type="torch")
        )
        return transformed_dataset

    def prepare_data(self):
        self.transform(
            self.dataset,
            self.tokenizer
        ).save_to_disk(self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            full_feats = load_from_disk(self.data_dir)
            self.train, self.val = full_feats.train_test_split(test_size=0.20).values()
        if stage == "test":
            full_feats = load_from_disk(self.data_dir)
            self.test = full_feats.shuffle().select(range(1_000))

    def train_dataloader(self):
        return DataLoader(self.train, sampler=RandomSampler(self.train), batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, sampler=RandomSampler(self.val), batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.test, sampler=RandomSampler(self.test), batch_size=32)
    
    def teardown(self, stage):
        shutil.rmtree(self.data_dir)