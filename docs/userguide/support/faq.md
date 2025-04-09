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

## Earth2Studio not authorized to download public NGC model checkpoints

Earth2Studio will attempt to use NGC CLI based authentication to download models.
Sometimes some misconfiguration can impact access to even public models with potential
errors like:

```bash
ValueError: Invalid org. Choose from ['no-team', '0123456789']
# or
ValueError: Invalid team. Choose from ['no-team', '0123456789']
```

In these cases it's typically because there is an NGC API key on the system either via
the NGC config file typically at `~/.ngc/config` or by environment variable
`NGC_CLI_API_KEY`.

One solution is to rename your config file or unset the API key environment variable so
Earth2Studio uses guest access.
Otherwise one can modify the config / environment variables to provide the needed
information.
For example:

```bash
export NGC_CLI_ORG=no-org
export NGC_CLI_TEAM=no-team
```

For more information see the [NGC CLI docs](https://docs.ngc.nvidia.com/cli/index.html).
