## Description

Add `ClassName` prognostic model for BRIEF_DESCRIPTION.

Closes #ISSUE_NUMBER (if applicable)

### Model details

| Property | Value |
|---|---|
| **Architecture** | PyTorch / ONNX / JAX |
| **Time step** | Xh (e.g., 6h, 24h) |
| **Input variables** | N variables (list or link) |
| **Output variables** | M variables (list or link) |
| **Spatial resolution** | X° x Y° (NxM grid) |
| **History required** | None / Xh (e.g., [-6h, 0h]) |
| **Checkpoint source** | NGC / HuggingFace / S3 |
| **Checkpoint size** | XX MB |
| **Checkpoint license** | LICENSE_NAME / Unknown |
| **Reference** | PAPER_OR_REPO_URL |

> **Note:** If the checkpoint/model weights license is not clearly
> documented in the source repository, mark it as "Unknown" and add a
> comment noting where you looked (e.g., "No LICENSE file in model repo,
> checked README and HuggingFace model card").

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `PACKAGE_NAME` | `>=X.Y` | LICENSE_TYPE | [link](LICENSE_URL) | REASON |

(or "No new dependencies needed")

When filling this table, look up each new dependency's license:

1. Check the package's PyPI page or repository for the license type
2. Link directly to the license file
3. Flag any **non-permissive licenses** (GPL, AGPL, SSPL) — these
   may be incompatible with the project's Apache-2.0 license

### Reference comparison

**Single step:**

- Max absolute difference: VALUE
- Max relative difference: VALUE
- Correlation: VALUE

**Multi-step (N steps):**

- Step 1: max_abs=VALUE, corr=VALUE
- Step N: max_abs=VALUE, corr=VALUE

### Validation

See sanity-check plots in PR comments below.

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues
