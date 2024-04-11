# Earth2Studio Contribution Guide

## Introduction

Thanks for your interest in contributing to Earth2Studio, this document provides a
brief overview of the essential items needed to add / modify code inside of
Earth2Studio. Your contribution will be a valued addition to the code base; we simply
ask that you read this page and understand our contribution process, whether you
are a seasoned open-source contributor or whether you are a first-time
contributor.

### Communicate with Us

We are happy to talk with you about your needs for Earth2Studio and your ideas for
contributing to the project. One way to do this is to create an issue discussing
your thoughts. It might be that a very similar feature is under development or
already exists, so an issue is a great starting point. If you are looking for an
issue to resolve that will help, refer to the
[issue](https://github.com/NVIDIA/Earth2Studio/issues) section.

We are happy to talk with you about your needs for Earth2Studio and your ideas for
contributing to the project. One way to do this is to create an issue discussing your
thoughts. It might be that a very similar feature is under development or already
exists, so an issue is a great starting point. If you are looking for an issue to
resolve that will help, refer to the [issue](https://github.com/NVIDIA/Earth2Studio/issues)
section.

## Contribute to Earth2Studio

### Pull Requests

Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo)
the [upstream](https://github.com/NVIDIA/Earth2Studio) Earth2Studio repository.

2. Git clone the forked repository and push changes to the personal fork.

3. Once the code changes are staged on the fork and ready for review, a
[Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR)
can be [requested](https://help.github.com/en/articles/creating-a-pull-request)
to merge the changes from a branch of the fork into a selected branch of upstream.

    - Exercise caution when selecting the source and target branches for the PR.
    - Ensure that you update the [`CHANGELOG.md`](CHANGELOG.md) to reflect your contributions.
    - Creation of a PR creation kicks off CI and a code review process.
    - Atleast one Earth2Studio engineer will be assigned for the review.

4. The PR will be accepted and the corresponding issue closed after adequate review and
testing has been completed. Note that every PR should correspond to an open issue and
should be linked on Github.

### Licensing Information

All source code files should start with this paragraph:

```bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the
contribution is your original work, or you have rights to submit it under the same
license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Adding a cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

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

### Pre-commit

For Earth2Studio development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline, but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

To install `pre-commit` run the following inside the Earth2Studio repository folder:

```bash
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and all
the commits will be checked for appropriate formatting.

### CI

To ensure quality of the code, your merge request will pass through several CI checks.
It is mandatory for your MRs to pass these pipelines to ensure a successful merge.
Please keep checking this document for the latest guidelines on pushing code.

### Documentation

To build the core documentation without executing examples use:

```bash
make docs
```

For full docs, where all examples are ran use:

```bash
make docs-full
```

Sometimes when developing an example you want to see what the end result looks like.
For development, its recommend to use the following process:

```bash
# Run this once
make docs

# Run this to generate your example
make docs-dev FILENAME=<example filename .py>
```

Build files will always be in `docs/_build/html`.
