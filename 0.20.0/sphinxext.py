# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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


def reset_torch(gallery_conf, fname):
    """Reset PyTorch's state between examples."""
    import numpy
    import torch

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Reset random seeds
    numpy.random.seed(42)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def reset_physicsnemo(gallery_conf, fname):
    """Reset PhysicsNemos's state between examples."""
    pass
    # import sys
    # Clear module for fresh imports
    # OKAY DONT DO THIS, THIS CAN CAUSE SOME ISSUES WITH ISINSTANCE AND STUFF
    # WHAT A WASTE OF A DAY
    # modules_to_clear = [mod for mod in sys.modules if mod.startswith("physicsnemo")]
    # for mod in modules_to_clear:
    #     sys.modules.pop(mod, None)
