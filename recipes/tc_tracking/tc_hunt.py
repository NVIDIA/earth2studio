import os

# Set MKL/OMP threading environment variables BEFORE any other imports
# This prevents MKL initialization race conditions that can cause
# divide-by-zero crashes when running FCN3 with NCCL/UCX
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from src.modes.generate_ensembles import generate_ensemble  # noqa: E402


@hydra.main(version_base=None, config_path="cfg", config_name="none")
def tc_hunt(cfg: DictConfig) -> None:
    """main function with initialisation."""

    if cfg.mode == "generate_ensemble":
        generate_ensemble(cfg)

    else:
        raise ValueError(f'invalid mode: {cfg.mode}, choose from "generate_ensemble"')

    print("finished **yaaayyyy**")

    return


if __name__ == "__main__":
    tc_hunt()
