# AutoModels { #automodel_userguide }

Earth2Studio offers a selection of pre-trained model checkpoints.
The fetching and caching of the model's checkpoint files is the responsibility of the
`earth2studio.models.auto.AutoModelMixin` and
`earth2studio.models.auto.Package` classes.
Understanding how these classes work can help users customize where model checkpoints
are stored as well as how to add their own pre-trained model to the package.

Model weights for Earth2Studio can be stored in a variety of locations that make them
publicly accessible to users.
When contributing a model, reach out with an issue to discuss the storage of such files.
Providing all checkpoint files is a *requirement* for any pre-trained model.
The following are the suggested locations in order of preference:

- *NGC Model Registry* - If a model is supported/developed by Nvidia PhysicsNeMo team, the
checkpoint can be uploaded on the [NGC model registry](https://catalog.ngc.nvidia.com/models).
This is the preferred location for Nvidia supported models that have gone under a more
rigorous internal evaluation. Private registries are also supported.

- *Huggingface Model Registry* - Huggingface offers
[model registries](https://huggingface.co/models) that any user can upload and share
checkpoint files with.
This is the method used for model files of several models that are not developed or
trained by Nvidia.

- *S3 Object Storage* - Providing model checkpoints with a S3 bucket is also another
viable object assuming the egress cost for downloads are covered by the owner.

## AutoModelMixin

The `earth2studio.models.auto.AutoModelMixin` class provides the interface
that pre-trained models in Earth2Studio use.
Any automodel in Earth2Studio needs to implement both the
`load_default_package` and `load_model` functions.

| Method | Purpose |
| --- | --- |
| `load_default_package()` | Return the default `Package` pointing at the model assets. |
| `load_model(package)` | Instantiate the model and load weights from the provided package. |

`load_default_package` is typically simple to implement, typically a single
line of code that creates the `earth2studio.models.auto.Package` object.
`load_model` does the heavy lifting, instantiating the model and loading the
pre-trained weights.
All pre-trained models in Earth2Studio implement these methods.
For example, have a look at the FourCastNet implementations:

```python
--8<-- "earth2studio/models/px/fcn.py:fcn-default-package"
```

```python
--8<-- "earth2studio/models/px/fcn.py:fcn-load-model"
```

!!! note
    The `load_default_package` doesn't perform any downloading.
    Rather, it creates a pointer to the directory the checkpoint files exist in, offering a
    primitive abstract filesystem.
    `load_model` triggers the download of any files when the path is accessed using
    `package.get("local/dir/to/file")`.

## Package

The `earth2studio.models.auto.Package` class is an abstract representation of
a storage location that contains some artifacts used to load a pre-trained model.
This class abstracts away the download and caching of files on the local machine.
Given that a supported remote store type is used, the use of the package class is as
follows:

```python
from earth2studio.models.auto import Package

# Instantiate package by pointing it to a remote folder
package = Package("ngc://models/nvidia/modulus/modulus_fcn@v0.2")

# Fetch a file from the remote store using the resolve method
cached_path_to_file = package.resolve("fcn.zip")

# Open a buffered reader of the file
opened_file = package.open("fcn.zip")
```

In this example, when calling resolve or open, the asset at `ngc://models/nvidia/modulus/modulus_fcn@v0.2/fcn.zip`
will be fetched and cached on the local machine.
A file buffer will then be returned pointing to the cached version of the file.
The cached path is a directory on the local file system, which can be configured using
environment variables.
See [Configuration](../about/install.md#configuration_userguide) section for details.

!!! note
    Earth2Studio file system uses [Fsspec](https://filesystem-spec.readthedocs.io/en/latest/)
    caching for files in packages.
    We encourage users that are interested in this type of utility to learn more about
    Fsspec and the specification it defines for advanced usage.
