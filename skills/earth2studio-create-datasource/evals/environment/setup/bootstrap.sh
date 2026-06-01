#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Harbor pre_agent_setup: create uv venv (Python 3.13) and sync Earth2Studio dev deps.
# Requires the repo at /workspace/repo (pass --copy-repo to astra-skill-eval evaluate).

set -euo pipefail

REPO_ROOT="${EARTH2STUDIO_ROOT:-/workspace/repo}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "e2s-eval-bootstrap: missing ${REPO_ROOT}/pyproject.toml" >&2
    echo "Re-run with: astra-skill-eval evaluate <skill> --copy-repo ..." >&2
    exit 1
fi

cd "${REPO_ROOT}"

export UV_LINK_MODE=copy
export UV_PYTHON=3.13

uv venv --python 3.13
uv sync --group dev --extra data
uv run pre-commit install --install-hooks

cat >/etc/profile.d/e2s-eval.sh <<EOF
export EARTH2STUDIO_ROOT=${REPO_ROOT}
export PATH="${REPO_ROOT}/.venv/bin:\${PATH}"
cd ${REPO_ROOT}
EOF

echo "e2s-eval-bootstrap: ready at ${REPO_ROOT} ($(uv run python --version))"
