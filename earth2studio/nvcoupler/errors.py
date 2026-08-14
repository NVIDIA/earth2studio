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

"""Error hierarchy for the nvcoupler coupling framework.

Every validation error names the components and fields involved and, where
possible, the concrete fix — so a misconfigured coupled system fails at
initialize time with an actionable message rather than mid-rollout.
"""

import difflib
from collections.abc import Iterable


def suggest(name: str, candidates: Iterable[str], n: int = 3) -> str:
    """Format a 'did you mean' suffix from close matches, or empty string."""
    matches = difflib.get_close_matches(name, list(candidates), n=n, cutoff=0.5)
    if not matches:
        return ""
    return f" Did you mean: {', '.join(repr(m) for m in matches)}?"


class CouplingError(Exception):
    """Base class for all nvcoupler configuration and runtime errors."""


class UnknownFieldError(CouplingError, KeyError):
    """A name could not be resolved in the field dictionary."""

    def __init__(self, name: str, candidates: Iterable[str]):
        msg = (
            f"Field name {name!r} is not a registered standard name or alias."
            + suggest(name, candidates)
            + " Register it with FieldDictionary.register(FieldEntry(...)) or "
            "add an alias with FieldDictionary.add_alias(...)."
        )
        # KeyError renders its arg with repr, so store the message on the
        # CouplingError side and pass it through once
        super().__init__(msg)

    def __str__(self) -> str:  # undo KeyError's quoting of the message
        return self.args[0]


class UnmatchedImportError(CouplingError):
    """A component advertises an import that no other component exports."""

    def __init__(self, component: str, field: str, available_exports: dict[str, list[str]]):
        exports_flat = [f for fields in available_exports.values() for f in fields]
        listing = (
            "; ".join(
                f"{comp} exports {', '.join(fields)}"
                for comp, fields in available_exports.items()
                if fields
            )
            or "no component exports anything"
        )
        super().__init__(
            f"Component {component!r} imports {field!r} but no component exports it."
            + suggest(field, exports_flat)
            + f" Available exports: {listing}."
            + " Add an alias, a Mediator producing the derived field, or a "
            "DataComponent supplying it from a data source."
        )


class UnitsMismatchError(CouplingError):
    """Matched fields disagree on units."""

    def __init__(self, field: str, src: str, src_units: str, dst: str, dst_units: str):
        super().__init__(
            f"Field {field!r}: {src!r} exports units {src_units!r} but {dst!r} "
            f"expects {dst_units!r}. Unit conversion is not performed in v1 — "
            "convert in a Mediator or align the FieldDictionary entries."
        )


class IncompatibleFieldError(CouplingError):
    """A connector could not reconcile matched fields (grid/units/vertical)."""


class VerticalMismatchError(CouplingError):
    """Source and destination vertical coordinates cannot be reconciled."""


class CadenceError(CouplingError):
    """Component or slot cadence does not align with the driver clock."""

    def __init__(self, what: str, interval: str, dt: str):
        super().__init__(
            f"{what} interval {interval} is not a positive multiple of the "
            f"driver clock dt {dt}. Choose a driver dt that divides every "
            "component timestep (typically their GCD)."
        )


class AmbiguousCouplingError(CouplingError):
    """couple() found more than one exporter for an imported field."""

    def __init__(self, field: str, importer: str, exporters: list[str]):
        super().__init__(
            f"Import {field!r} of component {importer!r} is exported by multiple "
            f"components: {', '.join(exporters)}. Auto-wiring cannot choose — "
            "build the Driver explicitly with Connector(src, dst, fields=[...]) "
            "or a run-sequence DSL."
        )


class SequenceError(CouplingError):
    """A run sequence references unknown names or is otherwise invalid."""
