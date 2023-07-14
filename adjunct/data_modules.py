import os
import lightning as pl
from transformers import AutoTokenizer
from datasets import (
    Dataset,
    load_dataset,
    ClassLabel,
    set_caching_enabled,
    load_from_disk,
)

from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Tuple
from uuid import uuid1

set_caching_enabled(False)

# TODO: Refactor for general sequence to label classification task
class ICD10DataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer_checkpoint: str, data_checkpoint: str = "./", limit: int = None
    ):
        super().__init__()
        self.val = None
        self.train = None
        self.test = None
        self.tokenizer, self.dataset = self.init_module(
            tokenizer_checkpoint, data_checkpoint
        )
        self.labeler = ClassLabel(
            names=list(self.dataset.to_pandas()["labels"].unique())
        )
        self.n_labels = len(self.labeler.names)
        self.data_dir = f"/tmp/{uuid1()}"

    @staticmethod
    def encode_data_elements(row, src_col, tokenizer):
        return tokenizer.batch_encode_plus(
            row[src_col],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    
    

    @staticmethod
    def with_labels(row, labeler):
        return dict(labels=labeler.str2int(row["labels"]))

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
    def transform(dataset, encode_fn, tokenizer, labels_fn, labeler):
        return (
            dataset.map(
                encode_fn,
                fn_kwargs=dict(src_col="line_data", tokenizer=tokenizer),
                batched=True,
            )
            .with_format(type="torch")
            .select_columns(["input_ids", "attention_mask", "labels"])
            .map(labels_fn, fn_kwargs=dict(labeler=labeler))
        )

    def prepare_data(self):
        self.transform(
            self.dataset,
            self.encode_data_elements,
            self.tokenizer,
            self.with_labels,
            self.labeler,
        ).save_to_disk(self.data_dir)

    def setup(self, stage: str):
        full_feats = load_from_disk(self.data_dir)
        if stage == "fit":
            self.train, self.val = full_feats.train_test_split(test_size=0.35).values()
        if stage == "test":
            self.test = full_feats.shuffle().select(range(1_000))

    def train_dataloader(self):
        return DataLoader(self.train, sampler=RandomSampler(self.train), batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, sampler=RandomSampler(self.test), batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.test, sampler=RandomSampler(self.test), batch_size=16)

    def teardown(self):
        os.remove(self.data_dir)


class ConstOCRDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer_checkpoint: str, data_checkpoint: str = "./", limit: int = None
    ):
        super().__init__()
        self.val = None
        self.train = None
        self.test = None
        self.tokenizer, self.dataset = self.init_module(
            tokenizer_checkpoint, data_checkpoint
        )
        self.labeler = ClassLabel(
            names=list(self.dataset.to_pandas()["labels"].unique())
        )
        self.n_labels = len(self.labeler.names)
        self.data_dir = f"/tmp/{uuid1()}"

    @staticmethod
    def encode_data_elements(row, src_col, tokenizer):
        return tokenizer.batch_encode_plus(
            row[src_col],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

    @staticmethod
    def with_labels(row, labeler):
        return dict(labels=labeler.str2int(row["labels"]))

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
    def transform(dataset, encode_fn, tokenizer, labels_fn, labeler):
        return (
            dataset.map(
                encode_fn,
                fn_kwargs=dict(src_col="line_data", tokenizer=tokenizer),
                batched=True,
            )
            .with_format(type="torch")
            .select_columns(["input_ids", "attention_mask", "labels"])
            .map(labels_fn, fn_kwargs=dict(labeler=labeler))
        )

    def prepare_data(self):
        self.transform(
            self.dataset,
            self.encode_data_elements,
            self.tokenizer,
            self.with_labels,
            self.labeler,
        ).save_to_disk(self.data_dir)

    def setup(self, stage: str):
        full_feats = load_from_disk(self.data_dir)
        if stage == "fit":
            self.train, self.val = full_feats.train_test_split(test_size=0.35).values()
        if stage == "test":
            self.test = full_feats.shuffle().select(range(1_000))

    def train_dataloader(self):
        return DataLoader(self.train, sampler=RandomSampler(self.train), batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, sampler=RandomSampler(self.val), batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.test, sampler=RandomSampler(self.test), batch_size=16)

    def teardown(self):
        os.remove(self.data_dir)
