# Deployment and Serving

Earth2Studio is a Python library for building and running weather and climate AI
workflows.
Deploying those workflows as a service, such as an inference server with a REST API,
job queue, object storage, and client, is handled by a separate project:
**[PhysicsNeMo Serve](https://github.com/NVIDIA/physicsnemo-serve)**.

For deployment, containers, server configuration, and the client, see the
[physicsnemo-serve repository](https://github.com/NVIDIA/physicsnemo-serve) and its
documentation.
Report issues and feature requests for serving and deployment there.

!!! note "Moved from Earth2Studio"

    The beta `earth2studio.serve` package, the `serve` install extra, and the
    top-level `serve/` directory (server, client, Dockerfile, and example workflows)
    have been removed from Earth2Studio and now live in PhysicsNeMo Serve.
    See the [migration checklist](https://github.com/NVIDIA/earth2studio/blob/main/MIGRATION.md#serving)
    for details.
