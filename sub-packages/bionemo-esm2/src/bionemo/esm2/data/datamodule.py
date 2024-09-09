# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from bionemo.core.data.resamplers import PRNGResampleDataset
from bionemo.core.utils import random_utils
from bionemo.esm2.data import dataset, tokenizer
from bionemo.llm.data import collate
from bionemo.llm.utils.datamodule_utils import infer_num_samples


class ESMDataModule(pl.LightningDataModule):
    """LightningDataModule wrapper of `ESMDataset`."""

    def __init__(
        self,
        train_cluster_path: str | os.PathLike,
        train_database_path: str | os.PathLike,
        valid_cluster_path: str | os.PathLike,
        valid_database_path: str | os.PathLike,
        seed: int | None = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,  # TODO(@jomitchell) can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        tokenizer: tokenizer.BioNeMoAutoTokenizer = tokenizer.get_tokenizer(),
        use_sample_and_shuffle_dataset: bool = False,
        dump_location: str | os.PathLike | None = None,
    ) -> None:
        """Initialize the ESMDataModule.

        Args:
            train_cluster_path: A path to the parquet files containing UniRef90 training clusters.
            train_database_path: A path to the sqlite file mapping UniRef90 cluster IDs to sequences.
            valid_cluster_path: A path to the parquet files containing UniRef50 validation clusters.
            valid_database_path: A path to the sqlite file mapping UniRef50 cluster IDs to sequences.
            seed: Input random seed. If None, initializes randomly. Defaults to 42.
            min_seq_length: Whether to pad sequences to a minimum length. If None, no extra padding is added. Defaults
                to None.
            max_seq_length: The maximum context length for the ESM transformer. Defaults to 1024.
            micro_batch_size: Passed to MegatronDataSampler. Defaults to 4.
            global_batch_size: Passed to MegatronDataSampler.. Defaults to 8.
            num_workers: The number of workers for the pytorch Dataloaders. Defaults to 10.
            persistent_workers: Whether to keep the workers alive between epochs. Defaults to True.
            pin_memory: Whether to pin GPU memory in the pytorch Dataloaders. Defaults to True.
            rampup_batch_size: Passed to MegatronDataSampler. Defaults to None.
            mask_prob: The overall chance of masking a token and having it appear in the loss fn. Defaults to 0.15.
            mask_token_prob: Percentage of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
            mask_random_prob: Percentage of masked tokens assigned to a random amino acid. Defaults to 0.1.
            tokenizer: The ESM2 tokenizer. Defaults to the one returned by `tokenizer.get_tokenizer()`.
        """
        super().__init__()
        self._train_cluster_path = train_cluster_path
        self._train_database_path = train_database_path
        self._valid_cluster_path = valid_cluster_path
        self._valid_database_path = valid_database_path
        self._seed = seed
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._mask_random_prob = mask_random_prob
        self._tokenizer = tokenizer
        self._use_sample_and_shuffle_dataset = use_sample_and_shuffle_dataset
        self._dump_location = dump_location

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="single",  # `MegatronPretrainingRandomSampler` from "cyclic" is failing.
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        """Setup the ESMDataModule.

        Args:
            stage: Unused.

        Raises:
            RuntimeError: If the trainer is not attached, or if the trainer's max_steps is not set.
        """
        del stage  # Unused.
        rng = np.random.default_rng(self._seed)

        if not hasattr(self, "trainer") or self.trainer is None:
            raise RuntimeError("Setup should be completed when trainer and config are attached.")

        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used "
                "in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )

        max_train_steps = self.trainer.max_steps
        if max_train_steps <= 0:
            raise RuntimeError("Please specify trainer.max_steps")

        # Create training dataset
        num_train_samples = int(
            max_train_steps * self.data_sampler.global_batch_size
        )  # training data requires upsampling (multiply by max_train_steps) on single MegatronPretrainingRandomSampler
        _train_ds = dataset.create_train_dataset(
            cluster_file=self._train_cluster_path,
            db_path=self._train_database_path,
            total_samples=num_train_samples,
            seed=random_utils.get_seed_from_rng(rng),
            max_seq_length=self._max_seq_length,
            mask_prob=self._mask_prob,
            mask_token_prob=self._mask_token_prob,
            mask_random_prob=self._mask_random_prob,
            tokenizer=self._tokenizer,
            dump_location=self._dump_location,
        )
        if self._use_sample_and_shuffle_dataset:
            self._train_ds = self._sample_and_shuffle_dataset(
                _train_ds, None, "train"
            )  # shuffle manually without cyclic MegatronPretrainingRandomSampler
        else:
            self._train_ds = _train_ds

        # Create validation dataset
        val_clusters = dataset.create_valid_clusters(self._valid_cluster_path)
        num_val_samples = infer_num_samples(
            limit_batches=self.trainer.limit_val_batches,
            num_samples_in_dataset=len(val_clusters),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="val",
        )
        _valid_ds = dataset.create_valid_dataset(
            clusters=self._valid_cluster_path,
            db_path=self._valid_database_path,
            total_samples=num_val_samples,
            seed=random_utils.get_seed_from_rng(rng),
            max_seq_length=self._max_seq_length,
            mask_prob=self._mask_prob,
            mask_token_prob=self._mask_token_prob,
            mask_random_prob=self._mask_random_prob,
            tokenizer=self._tokenizer,
            dump_location=self._dump_location,
        )
        if self._use_sample_and_shuffle_dataset:
            self._valid_ds = self._sample_and_shuffle_dataset(
                _valid_ds, None, "val"
            )  # shuffle manually without cyclic MegatronPretrainingRandomSampler
        else:
            self._valid_ds = _valid_ds

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

    def _create_dataloader(self, dataset, **kwargs) -> torch.utils.data.DataLoader:
        assert self._tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id."

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self._tokenizer.pad_token_id,
                min_length=self._min_seq_length,
                max_length=self._max_seq_length,
            ),
            **kwargs,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the dataloader for training data."""
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for validation data."""
        return self._create_dataloader(self._valid_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Raises a not implemented error."""
        raise NotImplementedError("No test dataset provided for ESM2")

    def _sample_and_shuffle_dataset(self, dataset: dataset.ESMMaskedResidueDataset, num_samples: int, stage: str):  # noqa: D417
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        return PRNGResampleDataset(
            dataset,
            num_samples=num_samples,
            seed=self._seed + len(stage),
        )
