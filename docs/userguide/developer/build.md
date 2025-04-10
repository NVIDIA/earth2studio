# Build Process

## Setting Version

To check the version of the current package:

```bash
uv run hatch version
```

To bump the version automatically the follow [hatch commands](https://hatch.pypa.io/1.9/version/)
can be used:

```bash
# To bump to next pre-release
uv run hatch version minor,a

# To bump to release candidate version
uv run hatch version rc

# To bump to release candidate version
uv run hatch version release
```

## Building Wheel

To build the wheel, use the [uv build](https://docs.astral.sh/uv/guides/package/)
command:

```bash
uv build
```
