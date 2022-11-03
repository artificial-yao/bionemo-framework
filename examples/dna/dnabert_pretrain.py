# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from omegaconf.omegaconf import OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging
from bionemo.model.dnabert import DNABERTModel
from bionemo.model.utils import setup_trainer

import numpy as np
import pytorch_lightning as ptl



@hydra_runner(config_path="conf", config_name="dnabert_config")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    np.random.seed(cfg.model.seed)
    ptl.seed_everything(cfg.model.seed)

    trainer = setup_trainer(cfg)
    model = DNABERTModel(cfg.model, trainer)

    trainer.fit(model)

if __name__ == '__main__':
    main()
