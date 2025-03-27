# Frequently Asked Questions

## There is a feature in the docs but not in the pip install package?

By default the docs are tracking the main branch, meaning that there is a chance this is
a new feature that will be a part of the next release.
Make sure you select your installed Earth2Studio version in the docs to see the list of
features present.
If you would like to use a new feature, install from source following the
{ref}`install_guide` guide.

## Will Earth2Studio add XYZ model?

Whether its you're own model or someone else's checkpoint, we are always interested in
making Earth2Studio as feature rich as possible for users.
Open an issue to discuss the model you're interested in using / integrating and we can
work out a plan to get it integrated.

## Earth2Studio requires X.Y.Z package or Python version, can I use another?

Earth2Studio has adopted the [scientific python](https://scientific-python.org/specs/)
spec 0 for minimum supported dependencies.
This mean adopting a common time-based policy for dropping dependencies to encourage the
use of modern Python and packages.
This helps improve matainance of the package and security posture.
This does not imply a strict requirement for all functionality and does not apply to
optional packages.
