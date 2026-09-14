# ---------- #
# Numerics

# ---------- #
import torch
import torch.nn.functional as F

# ---------- #
# credit


from dataclasses import dataclass
from typing import Sequence


@dataclass
class WindArtifactFilterConfig:
    activate: bool = True
    # which fields/levels to build the mask from and to apply to
    mask_level: int = 14
    target_levels: Sequence[int] = tuple(range(9, 21))
    target_vars: Sequence[str] = ("U", "V", "T", "Qtot")  # your current choice
    # detection & smoothing
    speed_threshold: float = 3.0193274566643846
    smooth_sigma: float = 1.0
    # Anisotropic data smoothing: the residual artifact is a meridional stripe that
    # oscillates in LONGITUDE (high zonal wavenumber, ~zero meridional wavenumber),
    # while the jet is zonally uniform and meridionally sharp. So smooth strongly in
    # the zonal (lon) direction to kill the stripe, and barely in the meridional (lat)
    # direction to preserve the jet's latitude profile. If either is None it falls back
    # to smooth_sigma (isotropic = original behavior).
    smooth_sigma_zonal: float = None
    smooth_sigma_meridional: float = None
    dilation_zonal: int = 13
    dilation_meridional: int = 5
    falloff_sigma: float = 4.0
    # If True, after low-pass smoothing inside the flagged region, rescale the smoothed
    # field so its mask-weighted RMS amplitude matches the original. This removes the
    # grid-scale wiggle (the smoothed field has no near-2dx energy) but gives a genuine
    # jet back the peak strength the Gaussian would otherwise shave off.
    preserve_amplitude: bool = False

    def validate(self):
        assert self.dilation_zonal > 0 and self.dilation_meridional > 0, "Dilations must be positive"
        assert self.smooth_sigma > 0 and self.falloff_sigma > 0, "Sigmas must be positive"
        for s in (self.smooth_sigma_zonal, self.smooth_sigma_meridional):
            assert s is None or s > 0, "Anisotropic smooth sigmas must be None or positive"
        assert isinstance(self.target_levels, (list, tuple)), "target_levels must be a sequence"
        assert isinstance(self.target_vars, (list, tuple)), "target_vars must be a sequence"


def load_wind_filter_config(conf) -> WindArtifactFilterConfig:
    # Optional block in YAML:
    # conf["postprocessing"]["wind_artifact_filter"] = {
    #   "activate": true,
    #   "mask_level": 14,
    #   "target_levels": [9, ..., 20],
    #   "target_vars": ["U","V","T","Qtot"],
    #   "speed_threshold": 3.0193,
    #   "smooth_sigma": 1.0,
    #   "dilation_zonal": 13,
    #   "dilation_meridional": 5,
    #   "falloff_sigma": 4.0
    # }

    pp = conf.get("postprocessing", {})
    raw = pp.get("wind_artifact_filter", {})
    cfg = WindArtifactFilterConfig(
        activate=raw.get("activate", True),
        mask_level=raw.get("mask_level", 14),
        target_levels=tuple(raw.get("target_levels", list(range(9, 21)))),
        target_vars=tuple(raw.get("target_vars", ["U", "V", "T", "Qtot"])),
        speed_threshold=raw.get("speed_threshold", 3.0193274566643846),
        smooth_sigma=raw.get("smooth_sigma", 1.0),
        smooth_sigma_zonal=raw.get("smooth_sigma_zonal", None),
        smooth_sigma_meridional=raw.get("smooth_sigma_meridional", None),
        dilation_zonal=raw.get("dilation_zonal", 13),
        dilation_meridional=raw.get("dilation_meridional", 5),
        falloff_sigma=raw.get("falloff_sigma", 4.0),
        preserve_amplitude=raw.get("preserve_amplitude", False),
    )
    cfg.validate()
    return cfg


def wind_filter(field, gaussian_2d, kernel_size, smooth_blend_mask, preserve_amplitude=False):
    """
    Apply wind filtering to a 2D field [height, width]

    If preserve_amplitude is True, the smoothed field is rescaled so its
    mask-weighted RMS amplitude matches the original before blending. This keeps the
    jet's strength (the part a low-pass would erode) while still removing the
    grid-scale wiggle, since the smoothed field carries no near-2dx energy to amplify.
    """
    # Ensure field is 4D for F.conv2d: [1, 1, height, width]
    if field.dim() == 2:
        field = field.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Ensure gaussian_2d has correct 4D shape for F.conv2d: [1, 1, kernel_h, kernel_w]
    if gaussian_2d.dim() == 2:
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    elif gaussian_2d.dim() == 3:
        gaussian_2d = gaussian_2d.unsqueeze(1)  # Add in_channels=1 dimension

    # Apply filtering. Derive padding from the kernel's own (possibly rectangular) shape
    # so anisotropic zonal/meridional kernels keep the field dimensions; kernel_size arg
    # is retained only for backward compatibility.
    k_lat, k_lon = gaussian_2d.shape[-2], gaussian_2d.shape[-1]
    field_smooth = F.conv2d(field, gaussian_2d, padding=(k_lat // 2, k_lon // 2))

    if preserve_amplitude:
        # Restore the mask-weighted RMS amplitude the low-pass removed: alpha scales the
        # smooth field so sum(w*smooth^2) == sum(w*field^2) over the blended region.
        # alpha >= 1 typically (smoothing reduces variance); the smooth field has no
        # grid-scale energy, so this lifts the jet peak back without re-injecting noise.
        w = smooth_blend_mask
        num = (w * field**2).sum()
        den = (w * field_smooth**2).sum()
        alpha = torch.sqrt(num / (den + 1e-12))
        alpha = torch.clamp(alpha, max=4.0)  # guard against pathological over-amplification
        field_smooth = alpha * field_smooth

    field_filtered = smooth_blend_mask * field_smooth + (1 - smooth_blend_mask) * field

    if squeeze_output:
        return field_filtered.squeeze()
    else:
        return field_filtered


def post_process_wind_artifacts_deprecated(x, conf, enable_filtering=True):
    """
    Apply wind artifact filtering post-processing to model state

    Args:
        x: Model state tensor to filter
        conf: Configuration dictionary containing variable info
        enable_filtering: Whether to apply filtering (for easy on/off)

    Returns:
        None (modifies x in-place)
    """
    if not enable_filtering:
        return

    try:
        # Extract configuration
        varname_upper = conf["data"]["variables"]
        levels = conf["model"]["levels"]

        # Apply wind artifact filtering with sensible defaults
        apply_wind_artifact_filter_to_tensor(
            x=x,
            varname_upper=varname_upper,
            levels_per_var=levels,
            mask_level=14,  # Good middle troposphere level
            target_levels=range(9, 21),  # Upper troposphere where jets occur
            target_vars=["U", "V", "T", "Qtot"],  # Wind and related fields
            speed_threshold=3.0193274566643846,  # tuned threshold
            smooth_sigma=1,  # Light smoothing
            dilation_zonal=13,  # Wide for jets
            dilation_meridional=5,  # Narrow for jets
            falloff_sigma=4.0,  # Smooth transitions
        )
    except Exception as e:
        print(f"Wind artifact filtering failed: {e}")
        # Continue without filtering rather than crash


def post_process_wind_artifacts(x, conf, enable_filtering=True):
    """
    Apply wind artifact filtering post-processing to model state (in-place).
    """
    if not enable_filtering:
        return

    try:
        wf_cfg = load_wind_filter_config(conf)
        if not wf_cfg.activate:
            return

        varname_upper = conf["data"]["variables"]
        levels = conf["model"]["levels"]

        apply_wind_artifact_filter_to_tensor(
            x=x,
            varname_upper=varname_upper,
            levels_per_var=levels,
            mask_level=wf_cfg.mask_level,
            target_levels=wf_cfg.target_levels,
            target_vars=wf_cfg.target_vars,
            speed_threshold=wf_cfg.speed_threshold,
            smooth_sigma=wf_cfg.smooth_sigma,
            dilation_zonal=wf_cfg.dilation_zonal,
            dilation_meridional=wf_cfg.dilation_meridional,
            falloff_sigma=wf_cfg.falloff_sigma,
            preserve_amplitude=wf_cfg.preserve_amplitude,
            smooth_sigma_zonal=wf_cfg.smooth_sigma_zonal,
            smooth_sigma_meridional=wf_cfg.smooth_sigma_meridional,
        )
    except Exception as e:
        print(f"Wind artifact filtering failed: {e}")


def apply_wind_artifact_filter_to_tensor(
    x,
    varname_upper,
    levels_per_var,
    mask_level=14,
    target_levels=range(10, 20),
    target_vars=["U", "V", "T", "Q"],
    speed_threshold=3.0193274566643846,
    smooth_sigma=1.5,
    dilation_zonal=15,
    dilation_meridional=5,
    falloff_sigma=4.0,
    preserve_amplitude=False,
    smooth_sigma_zonal=None,
    smooth_sigma_meridional=None,
):
    """
    Complete wind artifact filtering pipeline:
    1. Calculate mask from specified level of U,V winds
    2. Apply mask to target levels of target variables
    3. Modify original tensor x in-place

    Args:
        x: Original tensor to modify [batch, channels*levels, height, width]
        varname_upper: List of variable names in order
        levels_per_var: Number of levels per variable
        mask_level: Which level to use for mask calculation (default: 14)
        target_levels: Levels to apply filtering to (default: 10-19)
        target_vars: Variables to filter (default: ['U', 'V', 'T', 'Q'])
        ... other filtering parameters

    Returns:
        None (modifies x in-place)
    """

    # Step 1: Split tensor into variables
    channels = len(varname_upper)
    vars_split = []
    for i in range(channels):
        start = i * levels_per_var
        end = (i + 1) * levels_per_var
        vars_split.append(x[:, start:end, :, :])

    var_dict = {varname_upper[i]: vars_split[i] for i in range(channels)}

    # Step 2: Extract U,V at mask_level for mask calculation
    if "U" not in var_dict or "V" not in var_dict:
        raise ValueError("U and V winds required for mask calculation")

    u_mask_level = var_dict["U"][:, mask_level, :, :].squeeze()
    v_mask_level = var_dict["V"][:, mask_level, :, :].squeeze()

    # print(f"Calculating wind artifact mask from level {mask_level}")

    # Step 3: Calculate mask using simple_wind_artifact_filter
    # print('smooth sigma: ',smooth_sigma)
    _, _, gaussian_2d, kernel_size, smooth_blend_mask = simple_wind_artifact_filter(
        u_mask_level,
        v_mask_level,
        speed_threshold=speed_threshold,
        smooth_sigma=smooth_sigma,
        dilation_zonal=dilation_zonal,
        dilation_meridional=dilation_meridional,
        falloff_sigma=falloff_sigma,
        smooth_sigma_zonal=smooth_sigma_zonal,
        smooth_sigma_meridional=smooth_sigma_meridional,
    )

    # Step 4: Apply mask to target levels of target variables
    for var_name in target_vars:
        if var_name not in var_dict:
            print(f"Warning: {var_name} not found, skipping")
            continue

        # Find tensor position for this variable
        var_idx = varname_upper.index(var_name)
        start_idx = var_idx * levels_per_var

        # Apply filtering to each target level
        for level in target_levels:
            if level >= var_dict[var_name].shape[1]:
                print(f"Warning: Level {level} exceeds available levels for {var_name}")
                continue

            # Extract 2D slice
            level_slice = var_dict[var_name][:, level, :, :].squeeze()

            # Apply wind filter with pre-calculated mask
            filtered_slice = wind_filter(
                level_slice, gaussian_2d, kernel_size, smooth_blend_mask, preserve_amplitude=preserve_amplitude
            )

            # Put back into original tensor x (IN-PLACE modification)
            tensor_position = start_idx + level
            x[:, tensor_position, :, :] = filtered_slice.unsqueeze(0)

        # print(f"Applied filtering to {var_name}: levels {list(target_levels)}")

    # print("Wind artifact filtering complete - tensor x modified in-place")


def simple_wind_artifact_filter(
    u_wind, v_wind, speed_threshold=25.0, smooth_sigma=2.0, dilation_zonal=9, dilation_meridional=3, falloff_sigma=3.0,
    smooth_sigma_zonal=None, smooth_sigma_meridional=None,
):
    """
    Simple approach with ANISOTROPIC dilation - wider in zonal direction for jet-like features
    """
    if u_wind.dim() == 2:
        u_wind = u_wind.unsqueeze(0).unsqueeze(0)
        v_wind = v_wind.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    device = u_wind.device
    dtype = u_wind.dtype

    # Detection based on wind speed
    wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
    high_speed_mask = wind_speed > speed_threshold

    # ANISOTROPIC DILATION - wider in zonal (longitude) direction
    mask_float = high_speed_mask.float().to(dtype=dtype)

    # Create RECTANGULAR dilation kernel - wider than tall
    # dilation_meridional = latitude direction (height)
    # dilation_zonal = longitude direction (width)
    dilation_kernel = torch.ones(1, 1, dilation_meridional, dilation_zonal, device=device, dtype=dtype)

    # Dilate with anisotropic kernel
    padding_lat = dilation_meridional // 2
    padding_lon = dilation_zonal // 2
    expanded_mask_float = F.conv2d(mask_float, dilation_kernel, padding=(padding_lat, padding_lon))
    expanded_mask_float = torch.clamp(expanded_mask_float, 0, 1)

    # ANISOTROPIC FALLOFF - also wider in zonal direction
    falloff_kernel_size_lat = int(2 * falloff_sigma * 2 + 1)  # Narrower in lat
    falloff_kernel_size_lon = int(2 * falloff_sigma * 4 + 1)  # Wider in lon

    if falloff_kernel_size_lat % 2 == 0:
        falloff_kernel_size_lat += 1
    if falloff_kernel_size_lon % 2 == 0:
        falloff_kernel_size_lon += 1

    # Create anisotropic Gaussian falloff
    x_lat = torch.arange(falloff_kernel_size_lat, dtype=dtype, device=device)
    x_lat = x_lat - falloff_kernel_size_lat // 2
    gaussian_1d_lat = torch.exp(-0.5 * (x_lat / falloff_sigma) ** 2)
    gaussian_1d_lat = gaussian_1d_lat / gaussian_1d_lat.sum()

    x_lon = torch.arange(falloff_kernel_size_lon, dtype=dtype, device=device)
    x_lon = x_lon - falloff_kernel_size_lon // 2
    gaussian_1d_lon = torch.exp(-0.5 * (x_lon / (falloff_sigma * 2)) ** 2)  # Wider spread
    gaussian_1d_lon = gaussian_1d_lon / gaussian_1d_lon.sum()

    # 2D anisotropic falloff kernel
    gaussian_2d_falloff = gaussian_1d_lat.unsqueeze(1) * gaussian_1d_lon.unsqueeze(0)
    gaussian_2d_falloff = gaussian_2d_falloff.unsqueeze(0).unsqueeze(0)

    # Apply anisotropic falloff
    smooth_blend_mask = F.conv2d(
        expanded_mask_float, gaussian_2d_falloff, padding=(falloff_kernel_size_lat // 2, falloff_kernel_size_lon // 2)
    )

    # Data-smoothing kernel (separable Gaussian). Anisotropic by default-when-set:
    #   sig_lon (zonal, x) smooths along longitude -> kills the zonally-oscillating stripe
    #   sig_lat (meridional, y) smooths along latitude -> kept small to spare the jet profile
    # Falls back to isotropic smooth_sigma when the per-axis sigmas are None.
    sig_lat = smooth_sigma if smooth_sigma_meridional is None else smooth_sigma_meridional
    sig_lon = smooth_sigma if smooth_sigma_zonal is None else smooth_sigma_zonal

    def _gauss1d(sig):
        ks = int(2 * sig * 3 + 1)
        if ks % 2 == 0:
            ks += 1
        xx = torch.arange(ks, dtype=dtype, device=device) - ks // 2
        g = torch.exp(-0.5 * (xx / sig) ** 2)
        return g / g.sum(), ks

    g_lat, k_lat = _gauss1d(sig_lat)  # along latitude  (kernel height)
    g_lon, k_lon = _gauss1d(sig_lon)  # along longitude (kernel width)
    gaussian_2d = (g_lat.unsqueeze(1) * g_lon.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1,1,k_lat,k_lon]
    kernel_size = max(k_lat, k_lon)  # retained for backward-compat return value

    # Apply smoothing (asymmetric padding so the rectangular kernel keeps the field shape)
    pad = (k_lat // 2, k_lon // 2)
    u_smoothed = F.conv2d(u_wind, gaussian_2d, padding=pad)
    v_smoothed = F.conv2d(v_wind, gaussian_2d, padding=pad)

    # Smooth blending
    u_filtered = smooth_blend_mask * u_smoothed + (1 - smooth_blend_mask) * u_wind
    v_filtered = smooth_blend_mask * v_smoothed + (1 - smooth_blend_mask) * v_wind

    if squeeze_output:
        return u_filtered.squeeze(), v_filtered.squeeze(), gaussian_2d.squeeze(), kernel_size, smooth_blend_mask
    else:
        return u_filtered, v_filtered, gaussian_2d, kernel_size, smooth_blend_mask
