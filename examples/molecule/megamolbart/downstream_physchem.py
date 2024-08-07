# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data import PhysChemPreprocess
from bionemo.model.molecule.megamolbart import FineTuneMegaMolBART
from bionemo.model.utils import (
    setup_trainer,
)


@hydra_runner(config_path="conf", config_name="finetune_config")
def main(cfg) -> None:
    logging.info("\n\n************* Fintune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Do preprocessing if preprocess
    if cfg.do_preprocessing:
        logging.info("************** Starting Data PreProcessing ***********")
        PhysChemPreprocess().prepare_dataset(
            links_file=cfg.model.data.links_file, output_dir=cfg.model.data.preprocessed_data_path
        )

        if cfg.model.data.split_data:
            PhysChemPreprocess()._process_split(
                links_file=cfg.model.data.links_file,
                output_dir=cfg.model.data.preprocessed_data_path,
                test_frac=cfg.model.data.test_frac,
                val_frac=cfg.model.data.val_frac,
            )
        logging.info("************** Finished Data PreProcessing ***********")

    # Load model
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg
    trainer = setup_trainer(cfg, builder=None, reset_accumulate_grad_batches=False)
    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = FineTuneMegaMolBART.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = FineTuneMegaMolBART(cfg.model, trainer)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
    if cfg.do_testing:
        if "test" in cfg.model.data.dataset:
            trainer.limit_train_batches = 0
            trainer.limit_val_batches = 0
            trainer.fit(model)
            trainer.test(model, ckpt_path=None)
        else:
            raise UserWarning(
                "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
            )
        logging.info("************** Finished Testing ***********")


if __name__ == "__main__":
    main()
