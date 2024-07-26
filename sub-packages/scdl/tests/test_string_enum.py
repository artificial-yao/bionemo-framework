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


import pytest

from bionemo.data.scdl.util.string_enum import StringEnum


def test_string_enum():
    class E(StringEnum):
        x = "y"
        test = "pass"
        win = "lose"

    e = E()
    # Checking iteration over enum members
    assert e.x == "y"
    assert e.test == "pass"
    assert e.win == "lose"

    # Checking for attribute errors when accessing non-existent attributes
    with pytest.raises(AttributeError):
        getattr(e, "singlecell")

    # Checking for errors when trying to set attribute values
    with pytest.raises(AttributeError):
        setattr(e, "x", "zed")
