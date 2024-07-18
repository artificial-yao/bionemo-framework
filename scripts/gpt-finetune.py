import pathlib
from typing import List, Optional, TypedDict

import numpy as np
import pytorch_lightning as pl
import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.megatron_parallel import DataT
from nemo.lightning.pytorch.callbacks import ModelTransform
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.lightning.resume import AutoResume

# In pytorch_lightning 2.0 these are commented as being "any iterable or collection of iterables"
#  for now we'll use them incase the lightning type becomes something more specific in a future release.
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from bionemo.contrib.lightning import LossLoggingCallback


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional[TokenizerSpec] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "GPT2BPETokenizer")
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )
        # NOTE: the datasets and other distributed state is instantiated in `setup` rather than in `__init__` to support
        #  the different kinds of accellerators/strategies that lightning supports. This is a common pattern in lightning.

    def setup(self, stage: str = "") -> None:
        """See lightning documentation for more information on the stage and setup method. It is not required but
        if you want to be efficient about only initializing data that is needed in a particular stage you can do it here.
        According to the documentation valid values match the available calls to trainer.{fit,validate,test,predict},
        for example stage="fit". If we wanted to be fancy we could only initialize train/val during "fit". We could
        only instantiate "test" data during "test" etc.
        """
        self._train_ds = _MockGPTDataset(self.tokenizer, "train", self.num_train_samples, self.seq_length)
        self._validation_ds = _MockGPTDataset(self.tokenizer, "valid", self.num_val_samples, self.seq_length)
        self._test_ds = _MockGPTDataset(self.tokenizer, "test", self.num_test_samples, self.seq_length)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class GptDataItem(TypedDict):
    tokens: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor


class _MockGPTDataset(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
    ):
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.vocab_size = tokenizer.vocab_size
        self.length = num_samples
        self.seed = seed

        self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length))).unsqueeze(0)
        self.attention_mask = self.attention_mask < 0.5
        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> GptDataItem:
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        # Always return the same thing
        np_gen = np.random.default_rng(seed=(self.seed))
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64))
        labels = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64))

        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

    def _collate_fn(self, batch: DataT) -> DataT:
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

    def collate_fn(self, batch: DataT) -> DataT:
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


if __name__ == "__main__":
    devices, seq_length = 1, 2048
    ckpt_path = "/workspace/bionemo2/lightning_logs/version_9/ckpt/common.pt"
    log = NeMoLogger()
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.float32,
        enable_nemo_ckpt_io=False,  # True,
    )
    trainer = nl.Trainer(
        devices=devices,
        # TODO(@georgea) apparently this takes the steps from the loaded checkpoint?
        max_steps=20,
        accelerator="gpu",
        strategy=strategy,
        callbacks=[LossLoggingCallback()],
        # TODO(@jstjohn) See if we can get the example working with mixed precision
        # plugins=nl.MegatronMixedPrecision(precision="float32", amp_O2=False),
    )

    app_state = log.setup(
        trainer,
        # resume_if_exists=getattr(resume, "resume_if_exists", False),
        # task_config=getattr(train, "__io__", None),
    )

    # strategy.setup_megatron_parallel(trainer, )

    # model = llm.GPTModel.load_from_checkpoint(ckpt_path)
    # model = strategy.load_checkpoint(ckpt_path)
    gpt_config = llm.GPTConfig(
        num_layers=4,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        pipeline_dtype=torch.float32,
    )
    data_module = MockDataModule(seq_length=seq_length, global_batch_size=32)
    # (georgea) why do I need to configure a dataset before loading a model? can the tokenizer be removed? if so, can it be added later?
    model = llm.GPTModel(gpt_config, tokenizer=data_module.tokenizer)

    resume = AutoResume(path=pathlib.Path(ckpt_path).parent)

    resume.setup(model, trainer)

    peft = llm.peft.LoRA()

    model.model_transform = peft

    # TODO append modeltransorm callback?
    trainer.callbacks.append(ModelTransform())

    # TODO update a trained model with additional heads?

    trainer.fit(model, data_module)
