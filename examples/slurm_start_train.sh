#!/bin/bash
# readonly _data_root="/lustre/fsw/coreai_climate_earth2/datasets/cwb-diffusions"
# readonly _us_data_root="/lustre/fsw/coreai_climate_earth2/datasets/hrrr"
# readonly _corrdiff_root="/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir"
readonly _code_root="/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/asui/"
# readonly _cont_mounts="${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
# readonly _cont_mounts="/lustre:/lustre:ro,${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
# readonly _nsys_override="/lustre/fsw/portfolios/coreai/users/asui/nsight-systems-2025.2/pkg:/usr/local/cuda/NsightSystems-cli-2024.6.2:rw"
# readonly _cont_mounts="/lustre:/lustre:ro,${_code_root}:/code:rw,${_nsys_override}"
readonly _cont_mounts="/lustre:/lustre:rw,${_code_root}:/code:rw"



# readonly _cout_image="/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/nealp/pytest.sqsh"
readonly _cout_image="gitlab-master.nvidia.com/earth-2/image:latest"
# readonly _cout_image="/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/tge/climate_bottle/cbottle_release.sqsh"
# readonly _cout_image="gitlab-master.nvidia.com/earth-2/distill_cbottle/cbottle_fastgen:v3"
# readonly _cout_image="gitlab-master.nvidia.com/earth-2/distill_cbottle/cbottle_fastgen:latest"



readonly _cont_name='asuicorrdiff_interactive'
srun -A coreai_devtech_all\
        -N1\
        --gpus 1 \
        -p interactive\
        -t 03:59:00\
        -J coreai_climate_earth2-corrdiff:test\
        --ntasks-per-node=1\
    --container-image=${_cout_image}\
    --container-mounts=${_cont_mounts}\
    --container-name=${_cont_name}\
    --pty bash