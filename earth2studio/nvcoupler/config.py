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

"""YAML configuration: serialize/reconstruct a coupled system (NUOPC config analog).

Where NUOPC drivers read run sequences and field dictionaries from config
files, :func:`to_yaml` / :func:`from_yaml` round-trip an nvcoupler Driver
through a small YAML schema::

    clock:      {start, stop, dt}
    sequence: |                       # run-sequence DSL, verbatim
      @6h
        ...
    dictionary: [...]                 # only non-default FieldEntry items
    aliases: {alias: standard_name}   # add_alias() additions vs the default
    components:
      <name>: {class: <import.path>, kwargs: {...}}
    connectors: [{src, dst, fields, time_policy, fill}]

Only import-path-constructible components round-trip in v1: the ``class``
key must name a module-level class or factory callable that rebuilds the
component from ``kwargs`` alone. Components wrapping closures (a bare
CallableComponent) cannot be serialized unless they carry a ``yaml_spec``
attribute — a ``{"class": ..., "kwargs": ...}`` dict declaring how to
rebuild them. Model checkpoints referenced by load paths (e.g.
``{load: 'earth2studio.models.px.Persistence'}``) are out of scope for v1.
"""

import importlib
import inspect
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from .clock import Clock, as_timedelta, fmt_timedelta
from .connector import Connector
from .dictionary import DEFAULT_DICTIONARY, CellMethod, FieldDictionary, FieldEntry
from .driver import Driver
from .errors import CouplingError
from .mediator import AccumulationMediator

if TYPE_CHECKING:
    from .component import Component


class _LiteralDumper(yaml.SafeDumper):
    """SafeDumper rendering multi-line strings as literal blocks (|)."""


def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_representer)


def _sanitize(value: Any) -> Any:
    """Coerce numpy scalars/arrays and time types to YAML-safe primitives."""
    if isinstance(value, np.timedelta64):
        return fmt_timedelta(value)
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="s"))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Serialization (Driver -> YAML)
# ---------------------------------------------------------------------------
def _entry_to_dict(entry: FieldEntry, aliases: list[str]) -> dict:
    d: dict[str, Any] = {
        "standard_name": entry.standard_name,
        "canonical_units": entry.canonical_units,
        "description": entry.description,
        "aliases": sorted(aliases),
    }
    if entry.cell_method is not None:
        cm = entry.cell_method
        d["cell_method"] = {
            "base": cm.base,
            "method": cm.method,
            "window": fmt_timedelta(cm.window),
        }
    return d


def _custom_entries(components: dict[str, "Component"]) -> list[dict]:
    """FieldEntry items present in any component dictionary but not in (or
    differing from) the default dictionary."""
    out: dict[str, dict] = {}
    for comp in components.values():
        d = comp.dictionary
        for std in d.standard_names():
            entry = d.resolve(std)
            default = (
                DEFAULT_DICTIONARY.resolve(std) if std in DEFAULT_DICTIONARY else None
            )
            if entry == default:
                continue
            aliases = [a for a, s in d._aliases.items() if s == std and a != std]
            out[std] = _entry_to_dict(entry, aliases)
    return [out[k] for k in sorted(out)]


def _custom_aliases(components: dict[str, "Component"]) -> dict[str, str]:
    """Alias -> standard-name additions relative to the default dictionary.

    Aliases added via :meth:`FieldDictionary.add_alias` after registration
    leave the FieldEntry itself equal to the default, so they are invisible
    to :func:`_custom_entries`; serialize the alias-map delta explicitly.
    Aliases originating from a component's ``variable_aliases`` kwarg are
    skipped — they are rebuilt from the component spec itself.
    """
    out: dict[str, str] = {}
    for comp in components.values():
        raw_to_std = getattr(comp, "_raw_to_std", {})
        for alias, std in comp.dictionary._aliases.items():
            if alias == std:
                continue
            if DEFAULT_DICTIONARY._aliases.get(alias) == std:
                continue
            if raw_to_std.get(alias) == std:
                continue
            existing = out.get(alias)
            if existing is not None and existing != std:
                raise CouplingError(
                    f"Alias {alias!r} maps to {existing!r} in one component "
                    f"dictionary and to {std!r} in another; make the alias "
                    "consistent across components before serializing to YAML"
                )
            out[alias] = std
    return {a: out[a] for a in sorted(out)}


def _component_spec(name: str, comp: "Component") -> dict:
    spec = getattr(comp, "yaml_spec", None)
    if spec is not None:
        if not isinstance(spec, dict) or "class" not in spec:
            raise CouplingError(
                f"Component {name!r}: yaml_spec must be a dict with a 'class' "
                f"import path (and optional 'kwargs'), got {spec!r}"
            )
        return {
            "class": spec["class"],
            "kwargs": _sanitize(spec.get("kwargs", {})),
        }
    if isinstance(comp, AccumulationMediator):
        cls = type(comp)
        return {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "kwargs": {
                "name": comp.name,
                "fields": list(comp.methods),
                "window": fmt_timedelta(comp.timestep),
            },
        }
    raise CouplingError(
        f"Component {name!r} ({type(comp).__name__}) is not serializable: it "
        "wraps Python state (a closure or model object) that YAML cannot "
        "reconstruct. Set a yaml_spec attribute on the component — a dict "
        "{'class': '<import.path.to.class_or_factory>', 'kwargs': {...}} that "
        "rebuilds it — or construct this system in Python. Model components "
        "referenced by load paths are out of scope for YAML round-trips in v1."
    )


def to_yaml(driver: Driver, path: str | os.PathLike | None = None) -> str:
    """Serialize a Driver to YAML text (optionally also written to `path`).

    Raises
    ------
    CouplingError
        If any component is neither an AccumulationMediator nor carries a
        ``yaml_spec`` attribute describing how to rebuild it.
    """
    doc: dict[str, Any] = OrderedDict()
    doc["clock"] = {
        "start": str(np.datetime_as_string(driver.clock.start, unit="s")),
        "stop": str(np.datetime_as_string(driver.clock.stop, unit="s")),
        "dt": fmt_timedelta(driver.clock.dt),
    }
    doc["sequence"] = str(driver.sequence)
    entries = _custom_entries(driver.components)
    if entries:
        doc["dictionary"] = entries
    aliases = _custom_aliases(driver.components)
    if aliases:
        doc["aliases"] = aliases
    doc["components"] = {
        name: _component_spec(name, comp) for name, comp in driver.components.items()
    }
    connectors = []
    for conn in driver._connectors.values():
        item: dict[str, Any] = {
            "src": conn.src.name,
            "dst": conn.dst.name,
            "time_policy": conn.time_policy,
            "fill": conn.fill,
        }
        if conn._fields is not None:
            item["fields"] = list(conn._fields)
        connectors.append(item)
    if connectors:
        doc["connectors"] = connectors
    text = yaml.dump(
        dict(doc), Dumper=_LiteralDumper, sort_keys=False, default_flow_style=False
    )
    if path is not None:
        with open(path, "w") as f:
            f.write(text)
    return text


# ---------------------------------------------------------------------------
# Deserialization (YAML -> Driver)
# ---------------------------------------------------------------------------
def _resolve_import(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise CouplingError(
            f"Component class {path!r} is not a dotted import path "
            "(expected e.g. 'earth2studio.nvcoupler.mediator.TrailingAverageMediator')"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise CouplingError(
            f"Cannot import module {module_path!r} for component class {path!r}: {e}"
        ) from e
    try:
        return getattr(module, attr)
    except AttributeError:
        raise CouplingError(
            f"Module {module_path!r} has no attribute {attr!r} "
            f"(from component class {path!r})"
        ) from None


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _build_dictionary(items: list[dict]) -> FieldDictionary:
    dictionary = FieldDictionary(DEFAULT_DICTIONARY)
    for item in items:
        cm = None
        if item.get("cell_method"):
            raw = item["cell_method"]
            cm = CellMethod(
                base=raw["base"],
                method=raw["method"],
                window=as_timedelta(raw["window"]),
            )
        dictionary.register(
            FieldEntry(
                standard_name=item["standard_name"],
                canonical_units=item["canonical_units"],
                description=item.get("description", ""),
                aliases=frozenset(item.get("aliases", ())),
                cell_method=cm,
            )
        )
    return dictionary


def from_yaml(path_or_str: str | os.PathLike) -> Driver:
    """Build an (uninitialized) Driver from YAML text or a YAML file path.

    Components are reconstructed by importing each ``class`` path and calling
    it with ``kwargs``; call ``driver.initialize(ics)`` afterwards as usual.
    """
    source = str(path_or_str)
    if "\n" not in source and os.path.exists(source):
        with open(source) as f:
            source = f.read()
    doc = yaml.safe_load(source)
    if not isinstance(doc, dict):
        raise CouplingError(
            "YAML config must be a mapping with 'clock', 'sequence' and "
            f"'components' keys, got {type(doc).__name__}"
        )
    missing = [k for k in ("clock", "sequence", "components") if k not in doc]
    if missing:
        raise CouplingError(f"YAML config is missing required keys: {missing}")

    clock_cfg = doc["clock"]
    clock = Clock(clock_cfg["start"], clock_cfg["stop"], clock_cfg["dt"])

    dictionary = None
    if doc.get("dictionary"):
        dictionary = _build_dictionary(doc["dictionary"])
    if doc.get("aliases"):
        if dictionary is None:
            dictionary = FieldDictionary(DEFAULT_DICTIONARY)
        for alias, std in doc["aliases"].items():
            dictionary.add_alias(std, alias)

    components: dict[str, Component] = {}
    for name, spec in doc["components"].items():
        if not isinstance(spec, dict) or "class" not in spec:
            raise CouplingError(
                f"Component {name!r}: expected {{class: <import.path>, "
                f"kwargs: {{...}}}}, got {spec!r}"
            )
        factory = _resolve_import(spec["class"])
        kwargs = dict(spec.get("kwargs") or {})
        if (
            dictionary is not None
            and "dictionary" not in kwargs
            and _accepts_kwarg(factory, "dictionary")
        ):
            kwargs["dictionary"] = dictionary
        try:
            components[name] = factory(**kwargs)
        except CouplingError:
            raise
        except Exception as e:
            raise CouplingError(
                f"Component {name!r}: {spec['class']}(**{spec.get('kwargs', {})}) "
                f"failed: {e}"
            ) from e

    connectors: list[Connector] = []
    for item in doc.get("connectors") or []:
        src, dst = item["src"], item["dst"]
        for endpoint in (src, dst):
            if endpoint not in components:
                raise CouplingError(
                    f"Connector {src}->{dst}: {endpoint!r} is not a configured "
                    f"component; have {sorted(components)}"
                )
        connectors.append(
            Connector(
                components[src],
                components[dst],
                fields=item.get("fields"),
                time_policy=item.get("time_policy", "constant"),
                fill=item.get("fill", "none"),
            )
        )

    return Driver(
        components,
        doc["sequence"],
        clock,
        connectors=connectors or None,
    )
