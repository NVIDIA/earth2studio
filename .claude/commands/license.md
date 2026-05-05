Check that every Python source file carries the required SPDX Apache-2.0 license header:

```bash
make license
```

The expected header is:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

Report any files that are missing or have an incorrect header.
