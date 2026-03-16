# Lepton.AI deployment

## Using the Lepton.AI Dashboard

We will use the Lepton.AI dashboard to start the inference service.
Please refer to your onboarding instructions to get access to this dashboard.

The dashboard has an `Endpoints` tab on the top.
This is used to deploy long running services such as inference.

* Click on the `Endpoints` tab, then click on the `Create Endpoint` button on the right hand side.
* Choose the `Create from Container Image` option.
* Set an appropriate Endpoint name.
* Resource:
  * Choose the GPU option. Currently we only support x1 GPU, but this will change in the future.
  * Choose any preemption policy.
* Image Configuration:
  * Set your custom docker image, or use one of the prebuilt tags as appropriate.
  * Set server port to 8000 for the inference container, and 8888 for the jupyter container.
  * A registry auth might need to be created to access a private registry. If so, supply it here.
  * For the custom command, refer to the [Custom Command](#custom-command) section.
* Access Tokens:
  * If required, we can create a new access token for authorization.
  * If one is created, then it will need to be supplied while calling the REST APIs using the header
    `-H "Authorization: Bearer ${TOKEN}"`.
* Environment variables and secrets can be provided if necessary (e.g. WANDB_API_KEY).
* Storage:
  * The inference container expects a mount for `/outputs`. Set this in the `Mount Path`.
  * During onboarding, your project is provided with some NFS storage at a certain path.
    You can provide a sub-directory within this path in the `From path`.
  * Volume should be `lepton-shared-fs` or `amlfs`.
* Click `Create` to create this endpoint. Choose 1 replica.

Once the endpoint scales and is ready, you can start sending REST API requests to it.

### Custom Command

The docker image as built from the default Dockerfile comes preset with the command to run the
service.
If the default settings in `serve/server/conf/config.yaml` are fine, then you can leave this
section below blank.
If you wish to override certain settings with env vars or have some custom setup of your own,
then provide those here.

```bash
#!/bin/bash
<Additional custom setup if needed>
```

## Debugging and logs

We can click on the Endpoint -> Replicas to bring up some additional options.

* Clicking on `API` brings up an option to run the various REST APIs.
  For e.g. health check, or list inference requests, etc.
* Clicking on `Terminal` for the specific replica opens a Terminal into the container.
* Click on `Logs` shows a live stream of the current logs (slightly delayed).
