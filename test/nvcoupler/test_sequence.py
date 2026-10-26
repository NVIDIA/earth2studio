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

from earth2studio.nvcoupler import CouplingError, RunSequence, Slot, derive_sequence

CADENCES = {"atmos": "6h", "impact": "6h", "ocean": "48h"}


def test_derive_sequence_orders_fast_to_slow_and_by_dependency():
    sequence = derive_sequence(CADENCES, before=[("atmos", "impact")])
    assert [slot.models for slot in sequence.slots] == [("atmos", "impact"), ("ocean",)]
    assert sequence.models() == {"atmos", "impact", "ocean"}
    assert str(sequence) == "@6h\n  atmos\n  impact\n@2d\n  ocean"


def test_derive_sequence_reorders_same_cadence_dependency():
    sequence = derive_sequence({"b": "6h", "a": "6h"}, before=[("a", "b")])
    assert sequence.slots[0].models == ("a", "b")


@pytest.mark.parametrize(
    "cadences,before,match",
    [
        ({}, [], "without models"),
        ({"a": "6h"}, [("a", "zz")], "unknown model"),
        ({"a": "6h", "b": "6h"}, [("a", "b"), ("b", "a")], "cycle"),
        ({"a": "48h", "b": "6h"}, [("a", "b")], "slower source"),
    ],
)
def test_derive_sequence_rejects(cadences, before, match):
    with pytest.raises(CouplingError, match=match):
        derive_sequence(cadences, before)


def _sequence(*slots: tuple[str, tuple[str, ...]]) -> RunSequence:
    return RunSequence(tuple(Slot(interval, models) for interval, models in slots))


@pytest.mark.parametrize(
    "sequence,before,match",
    [
        (_sequence(("6h", ("atmos", "impact", "nope"))), [], "unknown model"),
        (
            _sequence(("6h", ("atmos", "impact", "atmos")), ("48h", ("ocean",))),
            [],
            "more than once",
        ),
        (_sequence(("6h", ("atmos", "impact"))), [], "never advances"),
        (_sequence(("6h", ("atmos", "impact")), ("12h", ("ocean",))), [], "cadence 2d"),
        (
            _sequence(("4h", ("atmos", "impact")), ("48h", ("ocean",))),
            [],
            "not a multiple",
        ),
        (
            _sequence(("6h", ("impact", "atmos")), ("48h", ("ocean",))),
            [("atmos", "impact")],
            "must advance before",
        ),
    ],
)
def test_validate_rejects(sequence, before, match):
    cadences = {
        name: np.timedelta64(int(value[:-1]), "h") for name, value in CADENCES.items()
    }
    with pytest.raises(CouplingError, match=match):
        sequence.validate(cadences, "6h", before)


def test_empty_slot_and_sequence_rejected():
    with pytest.raises(CouplingError, match="names no models"):
        Slot("6h", ())
    with pytest.raises(CouplingError, match="no slots"):
        RunSequence(())
