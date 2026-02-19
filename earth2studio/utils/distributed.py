# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Literal

import torch
from physicsnemo.distributed import DistributedManager
from torch.distributed import rpc


class DistributedInference:
    """Inference a model on remote GPUs.

    DistributedInference can be used to inference a model on one or more remote GPUs
    (i.e. GPUs on other ranks of the distributed environment). The user can pass data to the
    remote models by calling the DistributedInference object. The input is automatically
    queued and passed to the next available remote GPU. The calls are asynchronous and the
    results can be obtained by iterating over the `results` method.

    Parameters
    ----------
    model : Type[Callable]
        The model to initialize on remote GPUs.

        This must be implemented as a callable object that has, at a minimum, a `forward`
        method that takes a tensor of input data and returns a tensor of output data.

        It can also have an __init__ constructor; this is called on each remote process
        when the DistributedInference is instantiated. The constructor can be used
        to load the model on the remote GPU and for other initialization.

        The model can also have other methods that can be called remotely using the
        `call_func` method of DistributedInference. This can be used e.g. to get information
        from the remote models to the main process.
    *args :
        Positional arguments to pass to the model constructor.
    remote_ranks : list[int] | None, optional
        The ranks of the remote GPUs to initialize the model on. If not provided, the model
        will be initialized on all other ranks found in the distributed environment.
    **kwargs :
        Keyword arguments to pass to the model constructor.
    """

    @staticmethod
    def initialize() -> None:
        """Initialize the DistributedInference interface.

        This function must be called before instantiating any DistributedInference objects,
        typically at the beginning of an inference script.
        """
        DistributedManager.initialize()
        dist = DistributedManager()

        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

        # build device map
        local_device = str(dist.device)
        (local_device_type, local_device_id) = local_device.split(":")
        if local_device_type != "cuda":
            raise ValueError("Only CUDA devices are supported.")
        local_device_num = int(local_device_id)
        device_num_list = [
            torch.empty(1, dtype=torch.int64, device=dist.device)
            for _ in range(dist.world_size)
        ]
        # gather device numbers from each worker
        local_device_num = torch.tensor(
            [local_device_num], dtype=torch.int64, device=dist.device
        )
        torch.distributed.all_gather(device_num_list, local_device_num)

        for rank in range(dist.world_size):
            if rank == dist.rank:
                continue
            remote_device_num = int(device_num_list[rank][0])
            remote_device = f"cuda:{remote_device_num}"
            options.set_device_map(f"worker{rank}", {local_device: remote_device})

        rpc.init_rpc(
            f"worker{dist.rank}",
            rank=dist.rank,
            world_size=dist.world_size,
            rpc_backend_options=options,
        )

    @staticmethod
    def finalize() -> None:
        """Shut down the DistributedInference interface.

        This function must be called, typically at the end of an inference script,
        to ensure that the ranks hosting remote models do not shut down prematurely.
        """
        rpc.shutdown()

    def __init__(
        self,
        model: type,
        *args: Any,
        remote_ranks: list[int] | None = None,
        **kwargs: Any,
    ):
        self.dist = DistributedManager()
        if remote_ranks is None:  # select all other ranks
            remote_ranks = list(range(self.dist.world_size))
            del remote_ranks[self.dist.rank]
        self.remote_ranks = remote_ranks
        self.available_remotes: Queue[int] = Queue(len(remote_ranks))
        self.out_queue: Queue[Any] = Queue(len(remote_ranks))

        # initialize remote models
        self.remote_models = {
            rank: rpc.remote(f"worker{rank}", model, args=args, kwargs=kwargs)
            for rank in remote_ranks
        }
        # initialize queue of available remotes
        for rank in remote_ranks:
            self.available_remotes.put(rank)

    def call_func(
        self, func: str, *args: Any, rank: int | Literal["all"] = "all", **kwargs: Any
    ) -> Any:
        """Call a member function of the remote model.

        This can be used e.g. to get information from the model or to set parameters.

        Parameters
        ----------
        func : str
            The name of the member function to call.
        *args :
            Additional positional arguments to pass to the function.
        rank : int | Literal["all"], optional
            The rank of the remote GPU to call the function on. If "all", the function
            will be called on all remote GPUs.
        **kwargs :
            Additional keyword arguments to pass to the function.

        Returns
        -------
            The result of the function call. If `rank` is "all", a list of results from all
            remote GPUs is returned.
        """
        if rank == "all":
            result = [
                self.call_func(func, *args, rank=rank, **kwargs)
                for rank in self.remote_ranks
            ]
            return result

        rm = self.remote_models[rank]
        remote_func = getattr(rm.rpc_sync(), func)
        return remote_func(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Inference the remote model asynchronously.

        This will block until a remote model is available to accept the inputs.

        Parameters
        ----------
        *args :
            Positional arguments to pass to the model `forward` method.
        **kwargs :
            Keyword arguments to pass to the model `forward` method.
        """

        # get a remote model from the queue (will block until one is available)
        rank = self.available_remotes.get()
        rm = self.remote_models[rank]
        torch.cuda.synchronize(device=self.dist.device)
        task = rm.rpc_async().forward(*args, **kwargs)

        def callback(
            completed_task: torch.futures.Future,
        ) -> None:  # called when the inference is finished
            result = completed_task.value()
            torch.cuda.synchronize(
                device=self.dist.device
            )  # necessary to ensure result is usable
            self.out_queue.put(result)
            self.available_remotes.put(rank)

        task.then(callback)

    def wait(self) -> None:
        """Wait for all inference tasks to finish."""

        for _ in range(len(self.remote_ranks)):
            self.available_remotes.get()
        self.out_queue.put(None)  # signal that the inference is finished
        for rank in self.remote_ranks:
            self.available_remotes.put(rank)

    def results(self) -> Generator[Any, None, None]:
        """Get the results of the inference tasks.

        This method will yield results until all inference tasks have finished. The results
        may arrive out of order with respect to the inference calls.
        """
        while (result := self.out_queue.get()) is not None:
            yield result


def local_concurrent_pipeline(tasks: list[Callable]) -> None:
    """Run a list of tasks concurrently on the local machine.

    This can be used to set up different stages of a distributed inference pipeline.
    It will block until all tasks have finished.

    Parameters
    ----------
    tasks : list[Callable]
        A list of tasks to run concurrently.
    """
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        for task in tasks:
            executor.submit(task)
