import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.hens_run import run_inference
from src.hens_utilities import initialise


@hydra.main(version_base=None, config_path="cfg", config_name="helene")
def main(cfg: DictConfig) -> None:
    """Main Hydra function with instantiation"""

    DistributedManager.initialize()

    (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cyclone_tracker,
        data_source,
        output_coords_dict,
        base_random_seed,
        writer_executor,
        writer_threads,
    ) = initialise(cfg)

    run_inference(
        cfg,
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cyclone_tracker,
        data_source,
        output_coords_dict,
        base_random_seed,
        writer_executor,
        writer_threads,
    )


if __name__ == "__main__":
    main()
