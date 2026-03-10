# Style Guide

The Earth2Studio style guide is set of conventions we have adopted detailing how to
write code for this project. A consistent style and process makes the package much
easier to understand for users and ensure compliance requirements are met.

## Code Formatting

Earth2Studio uses [black](https://github.com/psf/black) as the code formatter.
All committed code is expected to be compliant with the version of black specified in
the [pyproject.toml](https://github.com/NVIDIA/earth2studio/blob/main/pyproject.toml).
Pre-commit will format your code for you or one can run `make format` to run black on
all files.

Additional Code formatting principles:

- Files are always lower-case reflecting, use shortened name of component they hold
- Class names are capitalized using pascal-case
- Function names are lower case using snake-case

## Linting

Linting is performed by [ruff](https://github.com/astral-sh/ruff) which includes many
processes to ensure code between files is as consistent as possible.
Ruff is configured in the projects [pyproject.toml](https://github.com/NVIDIA/earth2studio/blob/main/pyproject.toml).
Pre-commit will run linting for you or one can run `make lint` to run ruff on all files.

## Type Hints

Earth2Studio is a [MyPy](https://mypy-lang.org/) compliant package, meaning type hints
are required for all functions (parameters and return object).
MyPy is configured in the packages [pyproject.toml](https://github.com/NVIDIA/earth2studio/blob/main/pyproject.toml).
On top of requiring type hints, the following guidelines should be used:

- To help keep APIs a concise, several of the common types used through out
    the package are defined in `earth2studio/utils/type.py` which should be
    used when applicable.

```{literalinclude} ../../../earth2studio/utils/type.py
:lines: 17-
:language: python
```

- Earth2Studio is Python 3.10+, thus type hinting using generic objects should be used
    instead of the `typing` package. See [PEP 585](https://peps.python.org/pep-0585/)
    for details.

- Earth2Studio is Python 3.10+, thus the `|` operator should be used instead of `Union`
    and `Optional` type operators. See [PEP 604](https://peps.python.org/pep-0604/) for
    details.

:::{note}
The pre-commit hook [PyUpgrade](https://github.com/asottile/pyupgrade) will enforce
Python 3.10 type styles automatically.
:::

## Licensing Information

All source code files *must* start with this paragraph:

```bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
```

## Signing Your Work

We **require** that all contributors "sign-off" on their commits. This certifies that
the contribution is your original work, or you have rights to submit it under the same
license, or a compatible license. Any contribution which contains commits that are not
Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

```bash
git commit -s -m "Adding a cool feature."
```

This will append the following to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

:::{admonition} Full text of the DCO
:class: dropdown

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.
```

```text
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to
submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge,
is covered under an appropriate open source license and I have the right under that
license to submit that work with modifications, whether created in whole or in part
by me, under the same open source license (unless I am permitted to submit under a
different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified
(a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
that a record of the contribution (including all personal information I submit with
it, including my sign-off) is maintained indefinitely and may be redistributed
consistent with this project or the open source license(s) involved.

```

:::
