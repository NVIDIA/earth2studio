// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { resolveBaseBranch, findBaselineRun } = require('./testmon-baseline.cjs');

test('copied PRs use the actual target branch, including RC targets', async () => {
  for (const branch of ['main', '1.0.0-rc', '1.1.0-rc']) {
    const github = { rest: { pulls: { get: async (args) => {
      assert.equal(args.pull_number, 1188);
      return { data: { base: { ref: branch } } };
    } } } };
    assert.equal(await resolveBaseBranch(github, {}, 'pull-request/1188'), branch);
  }
});

test('invalid copied PR names fail instead of silently using main', async () => {
  await assert.rejects(resolveBaseBranch({}, {}, 'pull-request/not-a-number'));
});

test('artifact selection excludes expired and other-branch databases', async () => {
  const github = {
    rest: { actions: { listArtifactsForRepo: () => {} } },
    paginate: async (_method, args) => {
      assert.equal(args.name, 'testmon-1.0.0-rc-test-data');
      return [
        { expired: false, workflow_run: { id: 31, head_branch: 'pull-request/1188' } },
        { expired: true, workflow_run: { id: 30, head_branch: '1.0.0-rc' } },
        { expired: false, workflow_run: { id: 29, head_branch: 'main' } },
        { expired: false, workflow_run: { id: 28, head_branch: '1.0.0-rc' } },
      ];
    },
  };
  assert.equal(await findBaselineRun(github, {}, '1.0.0-rc', 'test-data'), '28');
});

test('missing RC baseline is a cold start', async () => {
  const github = {
    rest: { actions: { listArtifactsForRepo: () => {} } },
    paginate: async () => [],
  };
  assert.equal(await findBaselineRun(github, {}, '1.0.0-rc', 'test'), '');
});
