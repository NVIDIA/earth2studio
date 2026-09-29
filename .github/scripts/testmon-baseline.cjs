// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Copied PR workflows are push events, so github.base_ref is unavailable.
async function resolveBaseBranch(github, repo, refName) {
  const match = /^pull-request\/(\d+)$/.exec(refName);
  if (!match) throw new Error(`Invalid copied PR branch: ${refName}`);
  const { data } = await github.rest.pulls.get({
    ...repo,
    pull_number: Number(match[1]),
  });
  return data.base.ref;
}

async function findBaselineRun(github, repo, branch, toxEnv) {
  const artifacts = await github.paginate(github.rest.actions.listArtifactsForRepo, {
    ...repo,
    name: `testmon-${branch}-${toxEnv}`,
    per_page: 100,
  });
  // The API returns newest artifacts first. Only trust baseline-branch runs.
  const artifact = artifacts.find((item) =>
    !item.expired && item.workflow_run?.head_branch === branch
  );
  return artifact ? String(artifact.workflow_run.id) : '';
}

module.exports = { resolveBaseBranch, findBaselineRun };
