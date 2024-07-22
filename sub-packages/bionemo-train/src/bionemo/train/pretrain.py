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


# TODO(@mgreaves, @jstjohn, @jomitchell) Consider different abstractions for pretraining, inference, and fine-tuning and see
#  how they would address code duplication in the case of ESM2+Geneformer as well as a third hypothetical model that does
#  not share the same types/loaders, such as OpenFold. The design should be flexible enough to allow for those differeht
#  use cases and not hide too much complexity that a user would want to customize, while reducing code duplication
#  between scripts.

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Protocol, Union, get_args

import pytorch_lightning as pl
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.resume import AutoResume
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from torch.nn import functional as F

from bionemo.contrib.data.singlecell.preprocess import PreprocessResources
from bionemo.contrib.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.contrib.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger
from bionemo.train.lightning import LossLoggingCallback


def main(
    data_dir: Path,
    num_nodes: int,
    devices: int,
    seq_length: int,
    result_dir: Path,
    wandb_project: Optional[str],
    wandb_offline: bool,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    num_dataset_workers: int,
    specification: ModuleSpec,
    lr: float,
    micro_batch_size: int,
    cosine_rampup_frac: float,
    cosine_hold_frac: float,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    wandb_entity: str = "clara-discovery",
    create_tensorboard_logger: bool = False,
):
    """Train a Geneformer model on single cell data.

    Args:
        data_dir (Path): Base directory for the data.
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        wandb_project (Optional[str]): weights and biases project name
        wandb_offline (bool): if wandb should happen in offline mode
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save
            an updated checkpoint
        num_dataset_workers (int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        cosine_rampup_frac (float): fraction of steps at the beginning of the run to ramp up the learning rate
        cosine_hold_frac (float): fraction of steps to hold the minimum learning rate at the end of the run
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        wandb_entity (str): the group to use for the wandb run, sometimes called a team, could also be your username
        create_tensorboard_logger (bool): create the tensorboard logger


    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup train/test/val data paths
    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    test_data_path = data_dir / "test"

    # Setup the strategy and trainer
    pipeline_model_parallel_size = 1
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=False,
    )

    wandb_options: Optional[WandbLoggerOptions] = (
        None
        if wandb_project is None
        else WandbLoggerOptions(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            log_model=False,
        )
    )
    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=num_nodes,
        callbacks=[LossLoggingCallback(), RichModelSummary(max_depth=4), LearningRateMonitor()],
        plugins=nl.MegatronMixedPrecision(precision=precision, amp_O2=False),
    )

    # Preprocess the data to get the tokenizer and median dictionary
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    # Configure the data module and model
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * int(num_nodes * devices / pipeline_model_parallel_size),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    geneformer_config = BioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.relu,  # TODO(@jstjohn) check this
        qk_layernorm=True,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=False,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=True,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option=biobert_spec_option,
    )

    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = BioBertLightningModule(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                # TODO(@jstjohn) try decoupled_lr
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_kwargs=wandb_options,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        # FIXME @skothenhill this doesn't work yet, but this is probably close to what we are supposed to do
        resume=AutoResume(resume_if_exists=resume_if_exists, resume_ignore_no_checkpoint=True),
    )


@dataclass(frozen=True)
class WanDbConfig:
    project: str
    offline: bool
    entity: str
    log_model: bool = False

    def to_dict(self) -> dict:
        return dict(**self.__dict__)


@dataclass(frozen=True)
class PretrainConfig:
    data_dir: Path
    num_nodes: int
    devices: int
    result_dir: Path
    num_steps: int
    limit_val_batches: int
    val_check_interval: int
    num_dataset_workers: int
    lr: float
    micro_batch_size: int
    cosine_rampup_frac: float
    cosine_hold_frac: float
    experiment_name: str
    resume_if_exists: bool
    precision: PrecisionTypes
    amp_O2: bool = False
    create_tensorboard_logger: bool = False

    @property
    def train_data_path(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_data_path(self) -> Path:
        return self.data_dir / "val"

    @property
    def test_data_path(self) -> Path:
        return self.data_dir / "test"


@dataclass(frozen=True)
class MegatronStrategyConfig:
    pipeline_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1

    @property
    def ddp(self) -> Literal["megatron"]:
        return "megatron"


class Preprocessor(Protocol):
    def preprocess(self) -> PreprocessResources: ...


MakeOptimizer = Callable[[PretrainConfig], MegatronOptimizerModule]

MakePreprocessor = Callable[[PretrainConfig], Preprocessor]

MakeDataModule = Callable[[PretrainConfig, PreprocessResources], pl.LightningDataModule]

BionemoModelConfig = Any

MakeModelConfig = Callable[[PretrainConfig], BionemoModelConfig]


def refactored_main(
    pretrain_config: PretrainConfig,
    wandb_config: Optional[WanDbConfig],
    strategy_or_config: Union[MegatronStrategyConfig, nl.MegatronStrategy],
    make_preprocessor: MakePreprocessor,
    make_datamodule: MakeDataModule,
    make_model_config: MakeModelConfig,
    make_optimizer: MakeOptimizer,
    model_specification: ModuleSpec,
    training_callbacks: Optional[List[pl.Callback]] = None,
):
    """Train a Geneformer model on single cell data.

    Args:
        data_dir (Path): Base directory for the data.
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        wandb_project (Optional[str]): weights and biases project name
        wandb_offline (bool): if wandb should happen in offline mode
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save
            an updated checkpoint
        num_dataset_workers (int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        cosine_rampup_frac (float): fraction of steps at the beginning of the run to ramp up the learning rate
        cosine_hold_frac (float): fraction of steps to hold the minimum learning rate at the end of the run
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        wandb_entity (str): the group to use for the wandb run, sometimes called a team, could also be your username
        create_tensorboard_logger (bool): create the tensorboard logger
    """

    if not pretrain_config.data_dir.is_dir():
        raise ValueError(f"Expecting data directory to exist: {pretrain_config.data_dir}")

    # Create the result directory if it does not exist.
    pretrain_config.result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
    if isinstance(strategy_or_config, MegatronStrategyConfig):
        strategy: nl.MegatronStrategy = nl.MegatronStrategy(
            tensor_model_parallel_size=strategy_or_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=strategy_or_config.pipeline_model_parallel_size,
            ddp="megatron",
            find_unused_parameters=True,
            enable_nemo_ckpt_io=False,
        )
    else:
        strategy = strategy_or_config

    trainer = nl.Trainer(
        devices=pretrain_config.devices,
        max_steps=pretrain_config.num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=pretrain_config.limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=pretrain_config.val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=pretrain_config.num_nodes,
        callbacks=(
            [LossLoggingCallback(), RichModelSummary(max_depth=4), LearningRateMonitor()]
            if training_callbacks is None
            else training_callbacks
        ),
        plugins=nl.MegatronMixedPrecision(precision=pretrain_config.precision, amp_O2=pretrain_config.amp_O2),
    )

    # Preprocess the data to get the tokenizer and median dictionary
    preprocessor = make_preprocessor(pretrain_config)
    # preprocessor = GeneformerPreprocess(
    #     download_directory=train_data_path,
    #     medians_file_path=train_data_path / "medians.json",
    #     tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    # )
    preprocessed_resources = preprocessor.preprocess()
    match preprocessed_resources:
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            raise ValueError(f"Preprocessing failed! ({type(preprocessor)}): {preprocessor=}")

    # Configure the data module and model
    data = make_datamodule(pretrain_config, preprocessed_resources)
    # data = SingleCellDataModule(
    #     seq_length=preseq_length,
    #     tokenizer=tokenizer,
    #     train_dataset_path=train_data_path,
    #     val_dataset_path=val_data_path,
    #     test_dataset_path=test_data_path,
    #     random_token_prob=0.1,  # this is the incorrect setting we originally used.
    #     median_dict=median_dict,
    #     micro_batch_size=micro_batch_size,
    #     global_batch_size=micro_batch_size * int(num_nodes * devices / pipeline_model_parallel_size),
    #     # persistent workers is supported when num_dataset_workers > 0
    #     persistent_workers=num_dataset_workers > 0,
    #     pin_memory=False,
    #     num_workers=num_dataset_workers,
    # )

    geneformer_config = make_model_config(pretrain_config, precision, model_specification)
    # geneformer_config = BioBertConfig(
    #     num_layers=6,
    #     hidden_size=256,
    #     ffn_hidden_size=512,
    #     num_attention_heads=4,
    #     seq_length=seq_length,
    #     fp32_residual_connection=False,  # TODO(@jstjohn) check this
    #     hidden_dropout=0.02,
    #     init_method_std=0.02,
    #     kv_channels=None,
    #     apply_query_key_layer_scaling=True,
    #     make_vocab_size_divisible_by=128,
    #     masked_softmax_fusion=True,  # TODO(@jstjohn) check this
    #     fp16_lm_cross_entropy=False,
    #     params_dtype=get_autocast_dtype(precision),
    #     pipeline_dtype=get_autocast_dtype(precision),
    #     autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
    #     gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
    #     layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
    #     layernorm_epsilon=1.0e-12,
    #     activation_func=F.relu,  # TODO(@jstjohn) check this
    #     qk_layernorm=True,  # TODO(@jstjohn) check this
    #     apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
    #     bias_activation_fusion=False,  # TODO(@jstjohn) check this
    #     bias_dropout_fusion=True,  # TODO(@jstjohn) check this
    #     get_attention_mask_from_fusion=False,
    #     attention_dropout=0.1,
    #     share_embeddings_and_output_weights=True,
    #     enable_autocast=True,  # This has to be set to True if we use the mixed precision plugin
    #     biobert_spec_option=biobert_spec_option,
    # )

    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = BioBertLightningModule(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                # TODO(@jstjohn) try decoupled_lr
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=str(pretrain_config.result_dir.absolute()),
        name=pretrain_config.experiment_name,
        initialize_tensorboard_logger=pretrain_config.create_tensorboard_logger,
        wandb_kwargs=None if wandb_config is None else wandb_config.to_dict(),
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=(
            # FIXME @skothenhill this doesn't work yet, but this is probably close to what we are supposed to do
            AutoResume(resume_if_exists=pretrain_config.resume_if_exists, resume_ignore_no_checkpoint=True)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Geneformer with single cell data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the data base directory, for example this might be "
        "/workspace/bionemo2/data/cellxgene_2023-12-15_small",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-4,
        help="Learning rate for training. Default is 1e-4. With bigger global batches try 1e-3",
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    # FIXME (@skothenhill) figure out how checkpointing and resumption should work with the new nemo trainer
    parser.add_argument(
        "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument(
        "--experiment-name", type=str, required=False, default="geneformer", help="Name of the experiment."
    )
    parser.add_argument("--wandb-offline", action="store_true", default=False, help="Use wandb in offline mode.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=False,
        default=None,
        help="Wandb project name. Wandb will only happen if this is set..",
    )
    parser.add_argument(
        "--cosine-rampup-frac",
        type=float,
        required=False,
        default=0.01,
        help="Fraction of steps in which to ramp up the learning rate. Default is 0.01.",
    )
    parser.add_argument(
        "--cosine-hold-frac",
        type=float,
        required=False,
        default=0.05,
        help="Fraction of final steps in which to hold the minimum LR. Default is 0.05.",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of steps to use for training. Default is 0.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=False,
        default=2048,
        help="Sequence length of cell. Default is 2048.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        required=False,
        default=2,
        help="Number of steps to use for training. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--biobert-spec-option",
        type=BiobertSpecOption,
        choices=[e.value for e in BiobertSpecOption],
        required=False,
        default=BiobertSpecOption.bert_layer_local_spec.value,
        help="Biobert spec option to use for the model. Default is 'bert_layer_local_spec'.",
    )

    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        seq_length=args.seq_length,
        result_dir=args.result_dir,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_dataset_workers=args.num_dataset_workers,
        biobert_spec_option=args.biobert_spec_option,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        cosine_rampup_frac=args.cosine_rampup_frac,
        cosine_hold_frac=args.cosine_hold_frac,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
    )
