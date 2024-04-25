# Earth2Studio Contribution Guide

Thanks for your interest in contributing to Earth2Studio.
Your contribution will be a valued addition to the code base; we simply
ask that you read this page and understand our contribution process.

Please see the [developer guide](https://nvidia.github.io/earth2studio/userguide/developer/overview.html)
for more details.

## Pull Requests

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

## Issues

We use [GitHub issues](https://github.com/NVIDIA/earth2studio/issues) to track bugs,
feature requests and questions.
Please use the provided templates and ensure your description is clear and has
sufficient to reproduce the issue.

## License

The details of the license of Earth2Studio are detailed in the [LICENSE](./LICENSE)
file. By contributing you agree that your contributions will be licensed the same.
