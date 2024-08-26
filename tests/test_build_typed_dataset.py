# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pickle as pkl
import shutil

import pytest
from omegaconf import OmegaConf, open_dict

from bionemo.data.dataset_builder_utils import (
    _CSV_FIELDS_MMAP_TYPE,
    _CSV_MMAP_TYPE,
    _DATA_IMPL_TYPE_CLS,
    _FASTA_FIELDS_MMAP_TYPE,
    add_hash_to_metadata,
    build_typed_dataset,
)
from bionemo.data.mapped_dataset import ResamplingMappedDataset
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.tests import teardown_apex_megatron_cuda


def test_dataset_builder_fasta_fields():
    filepath = "examples/tests/test_data/preprocessing/test/uniref2022_small.fasta"
    data_impl = _FASTA_FIELDS_MMAP_TYPE
    kwargs = {"data_fields": {"id": 0, "sequence": 1}}

    cfg = {"data_impl_kwargs": {_FASTA_FIELDS_MMAP_TYPE: kwargs}}
    cfg = OmegaConf.create(cfg)
    dataset = build_typed_dataset(
        dataset_paths=filepath, data_impl=data_impl, use_upsampling=False, cfg=cfg, num_samples=None
    )

    assert isinstance(dataset, _DATA_IMPL_TYPE_CLS[data_impl]), (
        f"Dataset with data_imp={data_impl} should be " f"of class {_DATA_IMPL_TYPE_CLS[data_impl].__name__}"
    )

    assert len(dataset) == 100
    assert list(dataset[57].keys()) == ["n", "Tax", "TaxID", "RepID", "id", "sequence"]
    assert dataset[1]["n"] == "2"
    assert dataset[13]["Tax"] == "Galemys"
    assert dataset[28]["TaxID"] == "117571"
    assert dataset[81]["RepID"] == "A0A202DCN3_9BACT"
    assert dataset[64]["id"] == "UniRef50_A0A6F9DW11"
    seq = dataset[97]["sequence"]
    assert isinstance(seq, str)
    assert len(seq) == 24603
    assert (
        set(seq) - {"Y", "N", "W", "E", "F", "L", "A", "K", "R", "D", "C", "Q", "S", "H", "I", "V", "M", "G", "P", "T"}
    ) == set()


def test_add_hash_to_metadata():
    # Write file without SHA256
    info = {"newline_int": 10, "version": "0.2"}
    data_path = "examples/tests/test_data/molecule/test/x000.csv"
    idx_mapping_dir = "examples/tests/test_data/molecule/test/idx_dir"
    data_path_dir = os.path.dirname(data_path)
    os.makedirs(os.path.join(idx_mapping_dir, data_path_dir), exist_ok=True)
    pkl.dump(info, open(os.path.join(idx_mapping_dir, data_path + ".idx.info"), "wb"))

    # Add hash
    add_hash_to_metadata(idx_mapping_dir, [data_path])

    # Check that hash is present
    hash_info = pkl.load(open(os.path.join(idx_mapping_dir, data_path + ".idx.info"), "rb"))
    shutil.rmtree(idx_mapping_dir)  # clean up
    assert "sha256" in hash_info.keys(), "sha256 key not added to metadata file"


def test_dataset_builder_csv_mmap():
    # MegamolBART pretraining
    filepath = "examples/tests/test_data/molecule/test/x[000..001]"
    data_impl = _CSV_MMAP_TYPE
    kwargs = {
        "newline_int": 10,
        "header_lines": 1,
        "workers": 1,
        "sort_dataset_paths": True,
        "data_sep": ",",
        "data_col": 1,
    }
    cfg = {"data_impl_kwargs": {_CSV_MMAP_TYPE: kwargs}, "index_mapping_dir": ""}
    cfg = OmegaConf.create(cfg)

    dataset = build_typed_dataset(
        dataset_paths=filepath, data_impl=data_impl, use_upsampling=False, cfg=cfg, num_samples=None
    )
    assert len(dataset) == 18
    assert isinstance(dataset, _DATA_IMPL_TYPE_CLS[data_impl]), (
        f"Dataset with data_imp={data_impl} should be " f"of class {_DATA_IMPL_TYPE_CLS[data_impl].__name__}"
    )
    assert dataset[0] == "CN[C@H](CC(=O)NC1CC1)C(N)=O" and dataset[17] == "C=C[C@@H]1CCCCN1[C@@H]1CC[C@H]1N"


def test_dataset_builder_csv_fields_mmap():
    # MegaMolBART downstream task retrosynthesis
    filepath = "examples/tests/test_data/reaction/processed/test/data.csv"
    data_impl = _CSV_FIELDS_MMAP_TYPE
    kwargs = {
        "newline_int": 10,
        "header_lines": 1,
        "workers": 1,
        "sort_dataset_paths": False,
        "data_sep": ",",
        "data_fields": {"products": 3, "reactants": 2},
    }
    cfg = {"data_impl_kwargs": {_CSV_FIELDS_MMAP_TYPE: kwargs}}
    cfg = OmegaConf.create(cfg)
    dataset = build_typed_dataset(
        dataset_paths=filepath, data_impl=data_impl, use_upsampling=False, cfg=cfg, num_samples=None
    )
    assert len(dataset) == 10
    assert isinstance(dataset, _DATA_IMPL_TYPE_CLS[data_impl]), (
        f"Dataset with data_imp={data_impl} should be " f"of class {_DATA_IMPL_TYPE_CLS[data_impl].__name__}"
    )
    assert dataset[0] == {
        "products": "COC(=O)CCC(=O)c1ccc(OC2CCCCO2)cc1O",
        "reactants": "C1=COCCC1.COC(=O)CCC(=O)c1ccc(O)cc1O",
    }

    # testing if list will pass
    n = 2
    data_paths = ["examples/tests/test_data/reaction/processed/test/data.csv"] * n
    dataset_2 = build_typed_dataset(
        dataset_paths=data_paths, data_impl=data_impl, use_upsampling=False, cfg=cfg, num_samples=None
    )
    assert len(dataset_2) == (2 * len(dataset))


@pytest.mark.needs_gpu
def test_dataset_builder_upsampling():
    # MegaMolBART downstream task retrosynthesis
    filepath = "examples/tests/test_data/reaction/processed/test/data.csv"
    data_impl = _CSV_FIELDS_MMAP_TYPE
    kwargs = {
        "newline_int": 10,
        "header_lines": 1,
        "workers": 1,
        "sort_dataset_paths": False,
        "data_sep": ",",
        "data_fields": {"products": 3, "reactants": 2},
    }
    cfg = {"data_impl_kwargs": {_CSV_FIELDS_MMAP_TYPE: kwargs}}
    cfg = OmegaConf.create(cfg)
    with open_dict(cfg):
        cfg.max_seq_length = 512
        cfg.seed = 42

    # testing dataset upsampling - it requires initialize parallel distributed due to NeMoUpsampling
    initialize_distributed_parallel_state()
    num_samples = 1000
    dataset = build_typed_dataset(
        dataset_paths=filepath, data_impl=data_impl, use_upsampling=True, cfg=cfg, num_samples=num_samples
    )
    teardown_apex_megatron_cuda()
    assert len(dataset) == num_samples and len(dataset._dataset) == 10

    assert isinstance(dataset, ResamplingMappedDataset) and isinstance(
        dataset._dataset, _DATA_IMPL_TYPE_CLS[_CSV_FIELDS_MMAP_TYPE]
    )
