import os

# Set MKL/OMP threading environment variables BEFORE any other imports
# This prevents MKL initialization race conditions that cause
# divide-by-zero crashes when running FCN3 with NCCL/UCX
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import hydra
from omegaconf import DictConfig
from src.modes.baseline_extraction import extract_baseline
from src.modes.generate_ensembles import generate_ensemble, reproduce_members


@hydra.main(version_base=None, config_path="cfg", config_name="none")
def main(cfg: DictConfig) -> None:

    if cfg.mode == "extract_baseline":
        extract_baseline(cfg)

    elif cfg.mode == "generate_ensemble":
        generate_ensemble(cfg)

    elif cfg.mode == "reproduce_members":
        reproduce_members(cfg)

    else:
        raise ValueError(
            f'invalid mode: {cfg.mode}, choose from "extract_baseline" or "generate_ensemble"'
        )

    print(f"finished **yaaayyyy**")

    return


if __name__ == "__main__":
    main()
