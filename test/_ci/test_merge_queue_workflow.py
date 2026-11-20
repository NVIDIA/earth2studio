# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import json
import subprocess
import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/e2s-ci-test-pr.yml"
MANDATORY = ("changes", "format", "lint", "docs", "test", "coverage")
OPTIONAL = {
    "test-data": "data",
    "test-perturb": "perturb",
    "test-stats": "stats",
    "test-px-models": "px_models",
    "test-dx-models": "dx_models",
    "test-da-models": "da_models",
    "test-px-models-conflicts": "px_models",
}
HARNESS = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = {outputs: {}, errors: [], calls: []};
const core = {
  setOutput: (key, value) => { result.outputs[key] = value; },
  info: () => {},
  setFailed: message => result.errors.push(message),
};
const github = {rest: {pulls: {get: async args => {
  result.calls.push(args);
  return {data: {base: {ref: input.prBase || 'main'}}};
}}}};
process.env.CI_NEEDS = JSON.stringify(input.needs || {});
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  try {
    await new AsyncFunction('context', 'github', 'core', input.script)(
      input.context || {}, github, core);
  } catch (error) { result.errors.push(error.message); }
  process.stdout.write(JSON.stringify(result));
})();
"""


class MergeQueueWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # BaseLoader constructs only strings/containers and preserves the YAML "on" key.
        cls.workflow = yaml.load(
            WORKFLOW.read_text(),
            Loader=yaml.BaseLoader,  # noqa: S506
        )

    def execute(self, script, **inputs):
        # Execute the repository's scripts with a fixed harness and no shell.
        result = subprocess.run(  # noqa: S603
            ["node", "-e", HARNESS],  # noqa: S607
            input=json.dumps({"script": script, **inputs}),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def resolver(self, **inputs):
        steps = self.workflow["jobs"]["changes"]["steps"]
        script = next(step for step in steps if step.get("id") == "base")["with"][
            "script"
        ]
        return self.execute(script, **inputs)

    def gate(self, needs):
        self.assertTrue(
            "ci-passed" in self.workflow["jobs"], "CI passed gate is missing"
        )
        script = self.workflow["jobs"]["ci-passed"]["steps"][0]["with"]["script"]
        return self.execute(script, needs=needs)

    def needs(self, selected):
        needs = {name: {"result": "success"} for name in MANDATORY}
        needs["changes"]["outputs"] = {
            key: "true" if selected else "false" for key in OPTIONAL.values()
        }
        needs.update(
            {
                name: {"result": "success" if selected else "skipped"}
                for name in OPTIONAL
            }
        )
        return needs

    def test_workflow_contract(self):
        triggers = self.workflow["on"]
        self.assertIn("merge_group", triggers)
        self.assertEqual(triggers["merge_group"]["types"], ["checks_requested"])
        self.assertEqual(triggers["push"]["branches"], ["pull-request/**"])
        gate = self.workflow["jobs"]["ci-passed"]
        self.assertEqual(gate["name"], "CI passed")
        self.assertEqual(set(gate["needs"]), set(MANDATORY) | set(OPTIONAL))
        self.assertIn("always()", gate["if"])
        self.assertIn("github.repository == 'NVIDIA/earth2studio'", gate["if"])
        self.assertEqual(gate["steps"][0]["env"]["CI_NEEDS"], "${{ toJSON(needs) }}")

    def test_base_resolution(self):
        for branch in ("main", "1.1.0-rc"):
            with self.subTest(branch=branch):
                queue = self.resolver(
                    context={
                        "eventName": "merge_group",
                        "payload": {
                            "merge_group": {"base_ref": f"refs/heads/{branch}"}
                        },
                    }
                )
                self.assertEqual(queue["errors"], [])
                self.assertEqual(queue["outputs"]["branch"], branch)
                self.assertEqual(queue["calls"], [])
                pr = self.resolver(
                    prBase=branch,
                    context={
                        "eventName": "push",
                        "ref": "refs/heads/pull-request/42",
                        "repo": {"owner": "NVIDIA", "repo": "earth2studio"},
                    },
                )
                self.assertEqual(pr["outputs"]["branch"], branch)
                self.assertEqual(pr["errors"], [])
                self.assertEqual(pr["calls"][0]["pull_number"], 42)

    def test_invalid_base_resolution(self):
        for base in (None, "", "main", "refs/tags/main", "refs/heads/"):
            with self.subTest(base=base):
                result = self.resolver(
                    context={
                        "eventName": "merge_group",
                        "payload": {"merge_group": {"base_ref": base}},
                    }
                )
                self.assertTrue(result["errors"])
        self.assertTrue(
            self.resolver(context={"eventName": "merge_group", "payload": {}})["errors"]
        )
        self.assertTrue(
            self.resolver(context={"eventName": "push", "ref": "refs/heads/main"})[
                "errors"
            ]
        )

    def test_success_and_intentional_skips(self):
        for selected in (True, False):
            self.assertEqual(self.gate(self.needs(selected))["errors"], [])

    def test_required_jobs_cannot_fail_cancel_or_skip(self):
        for name in (*MANDATORY, *OPTIONAL):
            for state in ("failure", "cancelled", "skipped"):
                with self.subTest(name=name, state=state):
                    needs = self.needs(True)
                    needs[name]["result"] = state
                    self.assertTrue(self.gate(needs)["errors"])

    def test_unselected_failures_and_invalid_selection(self):
        for name, key in OPTIONAL.items():
            for state in ("failure", "cancelled"):
                with self.subTest(name=name, state=state):
                    needs = self.needs(False)
                    needs[name]["result"] = state
                    self.assertTrue(self.gate(needs)["errors"])
            for selection in (None, "", "invalid"):
                with self.subTest(name=name, selection=selection):
                    needs = self.needs(False)
                    if selection is None:
                        del needs["changes"]["outputs"][key]
                    else:
                        needs["changes"]["outputs"][key] = selection
                    self.assertTrue(self.gate(needs)["errors"])

    def test_missing_dependency_fails(self):
        for name in (*MANDATORY, *OPTIONAL):
            with self.subTest(name=name):
                needs = self.needs(True)
                del needs[name]
                self.assertTrue(self.gate(needs)["errors"])
