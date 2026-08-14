# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from earth2studio.nvcoupler.errors import CadenceError, SequenceError
from earth2studio.nvcoupler.mediator import TrailingAverageMediator
from earth2studio.nvcoupler.sequence import (
    ConnectAction,
    MediateAction,
    RunAction,
    RunSequence,
    Slot,
    parse_run_sequence,
)
from earth2studio.nvcoupler.testing import fake_atmos, fake_ocean

DSL = """
@6h
  atmos -> med          # accumulate into mediator
  ocean -> atmos        # lagged coupling
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""


def _components():
    return {
        "atmos": fake_atmos(),
        "ocean": fake_ocean(),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }


def test_parse_dsl():
    seq = parse_run_sequence(DSL)
    assert len(seq.slots) == 2
    assert seq.slots[0].interval == np.timedelta64(6, "h")
    assert seq.slots[0].actions == [
        ConnectAction("atmos", "med"),
        ConnectAction("ocean", "atmos"),
        RunAction("atmos"),
    ]
    assert seq.slots[1].actions == [
        MediateAction("med", "compute"),
        ConnectAction("med", "ocean"),
        RunAction("ocean"),
    ]


def test_parse_dsl_object_equivalence_and_str_roundtrip():
    seq = parse_run_sequence(DSL)
    manual = RunSequence(
        [
            Slot(
                "6h",
                [
                    ConnectAction("atmos", "med"),
                    ConnectAction("ocean", "atmos"),
                    RunAction("atmos"),
                ],
            ),
            Slot(
                "48h",
                [
                    MediateAction("med"),
                    ConnectAction("med", "ocean"),
                    RunAction("ocean"),
                ],
            ),
        ]
    )
    assert seq == manual
    assert parse_run_sequence(str(seq)) == seq  # str() emits valid DSL


def test_str_preserves_subhour_intervals():
    """'90m' and '30m' slots must not truncate to whole hours in str().

    Regression: str() used astype('timedelta64[h]'), silently turning a 90 m
    slot into '@1h' and a 30 m slot into '@0h' through YAML round-trips.
    """
    seq = parse_run_sequence("@90m\n  atmos\n@30m\n  ocean\n@")
    text = str(seq)
    assert "@90m" in text
    assert "@30m" in text
    rebuilt = parse_run_sequence(text)
    assert rebuilt == seq
    assert rebuilt.slots[0].interval == np.timedelta64(90, "m")
    assert rebuilt.slots[1].interval == np.timedelta64(30, "m")


def test_validate_mediator_cadence_mismatch():
    # med (48h window) scheduled via med.compute in the 6h slot must be
    # rejected, exactly like a RunAction cadence mismatch
    seq = parse_run_sequence(
        "@6h\n  atmos -> med\n  atmos\n  med.compute\n@48h\n  ocean\n@"
    )
    with pytest.raises(CadenceError, match="med"):
        seq.validate(_components(), "6h")


def test_parse_errors():
    with pytest.raises(SequenceError, match="outside any"):
        parse_run_sequence("atmos\n@6h\n")
    with pytest.raises(SequenceError, match="cannot parse"):
        parse_run_sequence("@6h\n  atmos -> \n")
    with pytest.raises(SequenceError, match="empty"):
        parse_run_sequence("# just a comment\n")


def test_validate_ok():
    seq = parse_run_sequence(DSL)
    seq.validate(_components(), "6h")


def test_validate_unknown_name_with_suggestion():
    seq = parse_run_sequence("@6h\n  atmoss\n@")
    with pytest.raises(SequenceError, match="atmos"):
        seq.validate(_components(), "6h")


def test_validate_cadence_mismatch():
    # ocean (48h) scheduled in the 6h slot
    seq = parse_run_sequence("@6h\n  atmos\n  ocean\n@48h\n  med.compute\n@")
    with pytest.raises(CadenceError):
        seq.validate(_components(), "6h")
    # slot interval not a multiple of driver dt
    seq = parse_run_sequence("@7h\n  atmos\n@")
    with pytest.raises(CadenceError):
        seq.validate(_components(), "6h")


def test_validate_idle_component():
    seq = parse_run_sequence("@6h\n  atmos\n@")
    with pytest.raises(SequenceError, match="never run"):
        seq.validate(_components(), "6h")


def test_helpers():
    seq = parse_run_sequence(DSL)
    assert seq.components_run() == {"atmos", "ocean", "med"}
    assert ConnectAction("ocean", "atmos") in seq.connections()
