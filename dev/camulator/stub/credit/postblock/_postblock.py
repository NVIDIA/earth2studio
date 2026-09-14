"""
postblock.py
-------------------------------------------------------
Content:
    - PostBlock
    - TracerFixer
    - GlobalMassFixer
    - GlobalWaterFixer
    - GlobalEnergyFixer

"""

import torch
from torch import nn

import numpy as np
import xarray as xr

from credit.data import get_forward_data
from credit.transforms import load_transforms
from credit.physics_core import physics_pressure_level, physics_hybrid_sigma_level
from credit.physics_constants import (
    GRAVITY,
    RHO_WATER,
    LH_WATER,
    CP_DRY,
    CP_VAPOR,
)
from credit.skebs import SKEBS

import logging
from math import pi

PI = pi
logger = logging.getLogger(__name__)


class PostBlock(nn.Module):
    def __init__(self, post_conf):
        """
        post_conf: dictionary with config options for PostBlock.
                   if post_conf is not specified in config,
                   defaults are set in the parser

        This class is a wrapper for all post-model operations.
        Registered modules:
            - SKEBS
            - TracerFixer
            - GlobalMassFixer
            - GlobalEnergyFixer
            - GlobalEnergyFixerUpDown

        """
        super().__init__()

        self.operations = nn.ModuleList()

        # The general order of postblock processes:
        # (1) tracer fixer --> mass fixer --> SKEB / water fixer --> energy fixer

        # negative tracer fixer
        if post_conf["tracer_fixer"]["activate"]:
            logger.info("TracerFixer registered")
            opt = TracerFixer(post_conf)
            self.operations.append(opt)

        # stochastic kinetic energy backscattering (SKEB)
        if post_conf["skebs"]["activate"]:
            logging.info("using SKEBS")
            self.operations.append(SKEBS(post_conf))

        # global mass fixer
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalMassFixer registered")
                opt = GlobalMassFixer(post_conf)
                self.operations.append(opt)

        # global water fixer
        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalWaterFixer registered")
                opt = GlobalWaterFixer(post_conf)
                self.operations.append(opt)

        # global energy fixer (net-flux version)
        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalEnergyFixer registered")
                opt = GlobalEnergyFixer(post_conf)
                self.operations.append(opt)

        # global energy fixer (up/down flux version)
        if post_conf.get("global_energy_fixer_updown", {}).get("activate", False):
            if post_conf["global_energy_fixer_updown"].get("activate_outside_model", False) is False:
                logger.info("GlobalEnergyFixerUpDown registered")
                opt = GlobalEnergyFixerUpDown(post_conf)
                self.operations.append(opt)

    def forward(self, x):
        for op in self.operations:
            x = op(x)

        if isinstance(x, dict):
            # if output is a dict, return y_pred (if it exists), otherwise return x
            return x.get("y_pred", x)
        else:
            # if output is not a dict (assuming tensor), return x
            return x


class TracerFixer(nn.Module):
    """
    This module fixes tracer values by replacing their values to a given threshold
    (e.g., `tracer[tracer<thres] = thres`).

    Args:
        post_conf (dict): config dictionary that includes all specs for the tracer fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        cfg = post_conf["tracer_fixer"]

        # ------------------------------------------------------------------------------ #
        # Build flat tracer_indices and per-channel thresholds.
        # Supports two config formats:
        #   (a) tracer_inds: [i0, i1, ...]  — flat list of channel indices (legacy)
        #   (b) tracer_ind_ranges: [[start, end], ...]  — inclusive ranges; thresholds
        #       are one-per-range and expanded to per-channel internally.
        #       tracer_var_names: ['Qtot', 'PRECT', ...]  — NC variable names for denorm.
        if "tracer_ind_ranges" in cfg:
            ranges = cfg["tracer_ind_ranges"]
            thres_per_range = cfg["tracer_thres"]
            thres_max_per_range = cfg.get("tracer_thres_max", None)

            self.tracer_indices = []
            self.tracer_thres = []
            # thres_max_per_range may be None (no cap) or a list that can contain null values.
            # Only activate per-channel max if any value is non-null.
            any_max = thres_max_per_range is not None and any(v is not None for v in thres_max_per_range)
            self.tracer_thres_max = [] if any_max else None

            for i, (start, end) in enumerate(ranges):
                n = end - start + 1
                self.tracer_indices.extend(range(start, end + 1))
                self.tracer_thres.extend([float(thres_per_range[i])] * n)
                if any_max:
                    v = thres_max_per_range[i]
                    self.tracer_thres_max.extend([float(v) if v is not None else float("inf")] * n)

            self._use_ranges = True
            self._ranges = ranges
            self._range_var_names = cfg.get("tracer_var_names", None)
        else:
            self.tracer_indices = cfg["tracer_inds"]
            self.tracer_thres = cfg["tracer_thres"]
            self.tracer_thres_max = cfg.get("tracer_thres_max", None)
            self._use_ranges = False
            self._range_var_names = cfg.get("tracer_var_names", None)

        # ------------------------------------------------------------------------------ #
        # Per-channel mean/std for denorm (loaded from NC files; NOT register_buffer).
        # Supports per-level 3-D variables (e.g. Qtot shape (32,)) and scalar 2-D vars.
        self.flag_denorm = bool(cfg.get("denorm", False))
        self.tracer_mean = None
        self.tracer_std = None

        if self.flag_denorm:
            mean_ds = xr.open_dataset(cfg["mean_path"]).load()
            std_ds = xr.open_dataset(cfg["std_path"]).load()

            chan_means = []
            chan_stds = []

            if self._use_ranges and self._range_var_names is not None:
                for i, (start, end) in enumerate(self._ranges):
                    n = end - start + 1
                    vname = self._range_var_names[i]
                    m = np.array(mean_ds[vname].values).flatten()
                    s = np.array(std_ds[vname].values).flatten()
                    if len(m) == 1:
                        chan_means.extend([float(m[0])] * n)
                        chan_stds.extend([float(s[0])] * n)
                    else:
                        chan_means.extend(m[:n].tolist())
                        chan_stds.extend(s[:n].tolist())
            elif self._range_var_names is not None:
                # flat tracer_inds with per-index var names (scalar stats assumed)
                for vname in self._range_var_names:
                    m = float(np.array(mean_ds[vname].values).flatten()[0])
                    s = float(np.array(std_ds[vname].values).flatten()[0])
                    chan_means.append(m)
                    chan_stds.append(s)
            else:
                logger.warning("TracerFixer: denorm=True but no tracer_var_names — stats default to 0/1")
                chan_means = [0.0] * len(self.tracer_indices)
                chan_stds = [1.0] * len(self.tracer_indices)

            # Stored as plain attributes (NOT register_buffer) → not in state_dict → EMA-safe
            self.tracer_mean = torch.tensor(chan_means).float()   # (n_tracers,)
            self.tracer_std = torch.tensor(chan_stds).float()

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get y_pred
        # y_pred is channel first: (batch, var, time, lat, lon)
        y_pred = x["y_pred"]
        orig_dtype = y_pred.dtype
        device = y_pred.device

        # ------------------------------------------------------------------------------ #
        # tracer correction — denorm per channel if requested, clip, renorm
        total_clipped_lo = 0
        total_clipped_hi = 0
        total_elements = 0

        if self.flag_denorm and self.tracer_mean is not None:
            for i, i_var in enumerate(self.tracer_indices):
                m = self.tracer_mean[i].to(device)
                s = self.tracer_std[i].to(device)

                # Inverse-normalize to physical units (float32 for precision)
                chan = y_pred[:, i_var, ...].float() * s + m

                # Clip (count before clip for diagnostics)
                thres = self.tracer_thres[i]
                total_clipped_lo += int((chan < thres).sum().item())
                chan[chan < thres] = thres
                if self.tracer_thres_max is not None:
                    thres_hi = self.tracer_thres_max[i]
                    if thres_hi is not None:
                        total_clipped_hi += int((chan >= thres_hi).sum().item())
                        chan[chan >= thres_hi] = thres_hi
                total_elements += chan.numel()

                # Re-normalize and write back in original dtype (in-place)
                y_pred[:, i_var, ...] = ((chan - m) / s).to(dtype=orig_dtype)
        else:
            # Normalized-space clipping (no denorm)
            for i, i_var in enumerate(self.tracer_indices):
                tracer_vals = y_pred[:, i_var, ...]
                thres = self.tracer_thres[i]
                total_clipped_lo += int((tracer_vals < thres).sum().item())
                tracer_vals[tracer_vals < thres] = thres
                if self.tracer_thres_max is not None:
                    thres_hi = self.tracer_thres_max[i]
                    total_clipped_hi += int((tracer_vals >= thres_hi).sum().item())
                    tracer_vals[tracer_vals >= thres_hi] = thres_hi
                total_elements += tracer_vals.numel()

        # ---- Diagnostic (every 50 steps) ----
        self._tracer_calls = getattr(self, "_tracer_calls", 0) + 1
        if self._tracer_calls % 50 == 1:
            pct_lo = 100.0 * total_clipped_lo / max(total_elements, 1)
            pct_hi = 100.0 * total_clipped_hi / max(total_elements, 1)
            logger.info(
                "TracerFixer step %d | clipped_lo=%d (%.4f%%)  clipped_hi=%d (%.4f%%)  "
                "total_elements=%d  n_channels=%d",
                self._tracer_calls,
                total_clipped_lo, pct_lo,
                total_clipped_hi, pct_hi,
                total_elements, len(self.tracer_indices),
            )

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalMassFixer(nn.Module):
    """
    This module applies global mass conservation fixes for both dry air and water budget.
    The output ensures that the global dry air mass and global water budgets are conserved
    through correction ratios applied during model runs. Variables `specific total water`
    and `precipitation` will be corrected to close the budget. All corrections are done
    using float32 PyTorch tensors.

    Args:
        post_conf (dict): config dictionary that includes all specs for the global mass fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_mass_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_mass_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(lon_demo, lat_demo, p_level_demo, midpoint=self.flag_midpoint)

            self.N_levels = len(p_level_demo)
            self.ind_fix = len(p_level_demo) - int(post_conf["global_mass_fixer"]["fix_level_num"]) + 1

        else:
            # the actual setup for model runs
            cfg_mass = post_conf["global_mass_fixer"]
            ds_physics = get_forward_data(cfg_mass["save_loc_physics"])

            lon_lat_level_names = cfg_mass["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = cfg_mass["midpoint"]

            if cfg_mass["grid_type"] == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)
                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, midpoint=self.flag_midpoint)
            # -------------------------------------------------------------------------- #
            self.ind_fix = self.N_levels - int(cfg_mass["fix_level_num"]) + 1

        # -------------------------------------------------------------------------- #
        if self.flag_midpoint:
            self.ind_fix_start = self.ind_fix
        else:
            self.ind_fix_start = self.ind_fix - 1

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf["global_mass_fixer"]["q_inds"][0])
        self.q_ind_end = int(post_conf["global_mass_fixer"]["q_inds"][-1]) + 1
        if self.flag_sigma_level:
            self.sp_ind = int(post_conf["global_mass_fixer"]["sp_inds"])

        # ------------------------------------------------------------------------------------ #
        # Per-variable mean/std for denorm (NOT register_buffer → EMA-safe).
        # Replaces load_transforms, which requires the old ERA5 single-source schema.
        cfg_mass = post_conf["global_mass_fixer"]
        self.flag_denorm_mass = bool(cfg_mass.get("denorm", False))
        self.q_mean_mass = None
        self.q_std_mass = None
        self.PS_mean_mass = None
        self.PS_std_mass = None

        if self.flag_denorm_mass:
            mean_ds = xr.open_dataset(cfg_mass["mean_path"]).load()
            std_ds = xr.open_dataset(cfg_mass["std_path"]).load()

            # Qtot: per-level stats (n_levels,) → (1, n_levels, 1, 1)
            q_m = torch.from_numpy(np.array(mean_ds["Qtot"].values)).float()
            q_s = torch.from_numpy(np.array(std_ds["Qtot"].values)).float()
            for d in [0, -1, -1]:
                q_m = q_m.unsqueeze(d)
                q_s = q_s.unsqueeze(d)
            self.q_mean_mass = q_m     # (1, n_levels, 1, 1)
            self.q_std_mass = q_s

            # PS: spatially varying (H, W) → (1, H, W)
            PS_m = torch.from_numpy(np.array(mean_ds["PS"].values)).float().unsqueeze(0)
            PS_s = torch.from_numpy(np.array(std_ds["PS"].values)).float().unsqueeze(0)
            self.PS_mean_mass = PS_m   # (1, H, W)
            self.PS_std_mass = PS_s

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        x_input = x["x"]
        y_pred = x["y_pred"]

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        N_vars = y_pred.shape[1]

        # Denorm only the variables needed (q and optionally sp) — float32 for precision
        device = y_pred.device
        orig_dtype_mass = y_pred.dtype

        if self.flag_denorm_mass:
            def _dn_q(t):
                return t.float() * self.q_std_mass.to(device) + self.q_mean_mass.to(device)

            def _rn_q(t):
                return (t.float() - self.q_mean_mass.to(device)) / self.q_std_mass.to(device)

            def _dn_sp(t):
                return t.float() * self.PS_std_mass.to(device) + self.PS_mean_mass.to(device)

            def _rn_sp(t):
                return (t.float() - self.PS_mean_mass.to(device)) / self.PS_std_mass.to(device)
        else:
            _dn_q = _rn_q = _dn_sp = _rn_sp = lambda t: t

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        # !!! Note: time dimension is collapsed throughout !!!

        q_input = _dn_q(x_input[:, self.q_ind_start : self.q_ind_end, -1, ...])
        q_pred = _dn_q(y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...])

        if self.flag_sigma_level:
            sp_input = _dn_sp(x_input[:, self.sp_ind, -1, ...])
            sp_pred = _dn_sp(y_pred[:, self.sp_ind, 0, ...])

        # ------------------------------------------------------------------------------ #
        # global dry air mass conservation

        if self.flag_sigma_level:
            # total dry air mass from q_input
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input, sp_input)

        else:
            # total dry air mass from q_input
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input)

            # total mass from q_pred
            mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(1 - q_pred, 0, self.ind_fix) / GRAVITY,
                axis=(-2, -1),
            )

            mass_dry_sum_t1_fix = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(1 - q_pred, self.ind_fix_start, self.N_levels) / GRAVITY,
                axis=(-2, -1),
            )

            q_correct_ratio = (mass_dry_sum_t0 - mass_dry_sum_t1_hold) / mass_dry_sum_t1_fix

            # broadcast: (batch, 1, 1, 1)
            q_correct_ratio = q_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # ===================================================================== #
            # q fixes based on the ratio
            # fix lower atmosphere
            q_pred_fix = 1 - (1 - q_pred[:, self.ind_fix_start :, ...]) * q_correct_ratio
            # extract unmodified part from q_pred
            q_pred_hold = q_pred[:, : self.ind_fix_start, ...]

            # concat upper and lower q vals
            # (batch, level, lat, lon)
            q_pred = torch.cat([q_pred_hold, q_pred_fix], dim=1)

            # ===================================================================== #
            # return fixed q back to y_pred

            # Renorm q back to normalized units, splice into y_pred
            q_pred_rn = _rn_q(q_pred).to(dtype=orig_dtype_mass).unsqueeze(2)
            y_pred = concat_fix(y_pred, q_pred_rn, self.q_ind_start, self.q_ind_end, N_vars)

        # ===================================================================== #
        # surface pressure fixes on global dry air mass conservation
        # model level only

        if self.flag_sigma_level:
            delta_coef_a = self.coef_a.diff().to(q_pred.device)
            delta_coef_b = self.coef_b.diff().to(q_pred.device)

            if self.flag_midpoint:
                p_dry_a = ((delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)).sum(1)
                p_dry_b = ((delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)).sum(1)
            else:
                q_mid = (q_pred[:, :-1, ...] + q_pred[:, 1:, ...]) / 2
                p_dry_a = ((delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)).sum(1)
                p_dry_b = ((delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)).sum(1)

            grid_area = self.core_compute.area.unsqueeze(0).to(q_pred.device)
            mass_dry_a = (p_dry_a * grid_area).sum((-2, -1)) / GRAVITY
            mass_dry_b = (p_dry_b * sp_pred * grid_area).sum((-2, -1)) / GRAVITY

            # sp correction ratio using t0 dry air mass and t1 moisture
            sp_correct_ratio = (mass_dry_sum_t0 - mass_dry_a) / mass_dry_b

            # ---- Diagnostic (every 50 steps) ----
            self._mass_calls = getattr(self, "_mass_calls", 0) + 1
            if self._mass_calls % 50 == 1:
                m0 = mass_dry_sum_t0.detach().float().mean().item()
                sp_r = sp_correct_ratio.detach().float()
                logger.info(
                    "MassFixer step %d | dry_mass_t0=%.4e  "
                    "sp_ratio mean=%.6f  min=%.6f  max=%.6f  "
                    "drift_pct=%.4f%%",
                    self._mass_calls,
                    m0,
                    sp_r.mean().item(), sp_r.min().item(), sp_r.max().item(),
                    100 * (sp_r.mean().item() - 1.0),
                )

            sp_correct_ratio = sp_correct_ratio.unsqueeze(1).unsqueeze(2)
            sp_pred = sp_pred * sp_correct_ratio

            # Renorm sp, splice into y_pred
            sp_pred_rn = _rn_sp(sp_pred).to(dtype=orig_dtype_mass).unsqueeze(1).unsqueeze(2)
            y_pred = concat_fix(y_pred, sp_pred_rn, self.sp_ind, self.sp_ind, N_vars)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalWaterFixer(nn.Module):
    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_water_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_water_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(lon_demo, lat_demo, p_level_demo, midpoint=self.flag_midpoint)
            self.N_levels = len(p_level_demo)
            cfg_water = post_conf["global_water_fixer"]
            self.N_seconds = int(cfg_water["lead_time_periods"]) * 3600

        else:
            # the actual setup for model runs
            cfg_water = post_conf["global_water_fixer"]
            ds_physics = get_forward_data(cfg_water["save_loc_physics"])

            # Water fixer shares the same statics file + grid type as mass fixer,
            # but reads its own config keys (fall back to global_mass_fixer if absent).
            lon_lat_level_names = cfg_water.get(
                "lon_lat_level_name",
                post_conf["global_mass_fixer"]["lon_lat_level_name"],
            )
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = cfg_water.get(
                "midpoint", post_conf["global_mass_fixer"]["midpoint"]
            )
            grid_type = cfg_water.get(
                "grid_type", post_conf["global_mass_fixer"]["grid_type"]
            )

            if grid_type == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)

                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, midpoint=self.flag_midpoint)

            self.N_seconds = int(cfg_water["lead_time_periods"]) * 3600

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        cfg_water = post_conf["global_water_fixer"]
        self.q_ind_start = int(cfg_water["q_inds"][0])
        self.q_ind_end = int(cfg_water["q_inds"][-1]) + 1
        self.precip_ind = int(cfg_water["precip_ind"])
        self.evapor_ind = int(cfg_water["evapor_ind"])
        if self.flag_sigma_level:
            self.sp_ind = int(cfg_water["sp_inds"])

        # ------------------------------------------------------------------------------------ #
        # Per-variable mean/std for denorm (NOT register_buffer → EMA-safe).
        self.flag_denorm_water = bool(cfg_water.get("denorm", False))
        self.q_mean_water = None
        self.q_std_water = None
        self.PS_mean_water = None
        self.PS_std_water = None
        self.precip_mean_water = None
        self.precip_std_water = None
        self.evapor_mean_water = None
        self.evapor_std_water = None

        if self.flag_denorm_water:
            mean_ds = xr.open_dataset(cfg_water["mean_path"]).load()
            std_ds = xr.open_dataset(cfg_water["std_path"]).load()

            def _w_buf(vname, expand_dims=None):
                m = torch.from_numpy(np.array(mean_ds[vname].values)).float()
                s = torch.from_numpy(np.array(std_ds[vname].values)).float()
                if expand_dims:
                    for d in expand_dims:
                        m = m.unsqueeze(d)
                        s = s.unsqueeze(d)
                return m, s

            q_m, q_s = _w_buf("Qtot", expand_dims=[0, -1, -1])  # (1, n_lev, 1, 1)
            PS_m, PS_s = _w_buf("PS", expand_dims=[0])            # (1, H, W)
            # Precipitation variable name may differ; use config key if provided
            precip_vname = cfg_water.get("precip_var_name", "PRECT")
            evapor_vname = cfg_water.get("evapor_var_name", "QFLX")
            prec_m, prec_s = _w_buf(precip_vname)
            evap_m, evap_s = _w_buf(evapor_vname)

            self.q_mean_water = q_m;        self.q_std_water = q_s
            self.PS_mean_water = PS_m;      self.PS_std_water = PS_s
            self.precip_mean_water = prec_m; self.precip_std_water = prec_s
            self.evapor_mean_water = evap_m; self.evapor_std_water = evap_s

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x["x"]
        y_pred = x["y_pred"]

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        N_vars = y_pred.shape[1]
        device = y_pred.device
        orig_dtype_water = y_pred.dtype

        if self.flag_denorm_water:
            def _dn_q_w(t):
                return t.float() * self.q_std_water.to(device) + self.q_mean_water.to(device)

            def _dn_sp_w(t):
                return t.float() * self.PS_std_water.to(device) + self.PS_mean_water.to(device)

            def _dn_scalar(t, m, s):
                return t.float() * s.to(device) + m.to(device)

            def _rn_scalar(t, m, s):
                return (t.float() - m.to(device)) / s.to(device)
        else:
            _dn_q_w = lambda t: t
            _dn_sp_w = lambda t: t
            _dn_scalar = lambda t, m, s: t
            _rn_scalar = lambda t, m, s: t

        q_input = _dn_q_w(x_input[:, self.q_ind_start : self.q_ind_end, -1, ...])

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        q_pred = _dn_q_w(y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...])
        precip = _dn_scalar(y_pred[:, self.precip_ind, 0, ...], self.precip_mean_water, self.precip_std_water)
        evapor = _dn_scalar(y_pred[:, self.evapor_ind, 0, ...], self.evapor_mean_water, self.evapor_std_water)

        if self.flag_sigma_level:
            sp_input = _dn_sp_w(x_input[:, self.sp_ind, -1, ...])
            sp_pred = _dn_sp_w(y_pred[:, self.sp_ind, 0, ...])

        # ------------------------------------------------------------------------------ #
        # global water balance
        precip_flux = precip * RHO_WATER / self.N_seconds
        evapor_flux = evapor * RHO_WATER / self.N_seconds

        # total water content (batch, var, time, lat, lon)
        if self.flag_sigma_level:
            TWC_input = self.core_compute.total_column_water(q_input, sp_input)
            TWC_pred = self.core_compute.total_column_water(q_pred, sp_pred)
        else:
            TWC_input = self.core_compute.total_column_water(q_input)
            TWC_pred = self.core_compute.total_column_water(q_pred)

        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds

        # global sum of total water content tendency
        TWC_sum = self.core_compute.weighted_sum(dTWC_dt, axis=(-2, -1))

        # global evaporation source
        E_sum = self.core_compute.weighted_sum(evapor_flux, axis=(-2, -1))

        # global precip sink
        P_sum = self.core_compute.weighted_sum(precip_flux, axis=(-2, -1))

        # global water balance residual
        residual = -TWC_sum - E_sum - P_sum

        # compute correction ratio
        P_correct_ratio = (P_sum + residual) / P_sum
        # P_correct_ratio = torch.clamp(P_correct_ratio, min=0.9, max=1.1)

        # ---- Diagnostic (every 50 steps) ----
        self._water_calls = getattr(self, "_water_calls", 0) + 1
        if self._water_calls % 50 == 1:
            r = P_correct_ratio.detach().float()
            logger.info(
                "WaterFixer step %d | "
                "dTWC/dt=%.4e  E_src=%.4e  P_sink=%.4e  residual=%.4e  "
                "P_ratio mean=%.6f  min=%.6f  max=%.6f  drift_pct=%.4f%%",
                self._water_calls,
                TWC_sum.detach().float().mean().item(),
                E_sum.detach().float().mean().item(),
                P_sum.detach().float().mean().item(),
                residual.detach().float().mean().item(),
                r.mean().item(), r.min().item(), r.max().item(),
                100 * (r.mean().item() - 1.0),
            )

        # conservation penalty: fractional imbalance squared — trainer adds this to loss
        x["water_conservation_loss"] = (P_correct_ratio - 1.0).pow(2).mean()

        # broadcast: (batch_size, 1, 1, 1)
        P_correct_ratio = P_correct_ratio.unsqueeze(-1).unsqueeze(-1)

        # apply correction on precip
        precip = precip * P_correct_ratio

        # ===================================================================== #
        # return fixed precip back to y_pred (renorm if needed)
        precip_rn = _rn_scalar(precip, self.precip_mean_water, self.precip_std_water)
        precip_rn = precip_rn.to(dtype=orig_dtype_water).unsqueeze(1).unsqueeze(2)
        y_pred = concat_fix(y_pred, precip_rn, self.precip_ind, self.precip_ind, N_vars)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalEnergyFixer(nn.Module):
    """
    This module applys global energy conservation fixes. The output ensures that the global sum
    of total energy in the atmosphere is balanced by radiantion and energy fluxes at the top of
    the atmosphere and the surface. Variables `air temperature` will be modified to close the
    budget. All corrections are done using float32 Pytorch tensors.

    Args:
        post_conf (dict): config dictionary that includes all specs for the global energy fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_energy_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_energy_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(
                lon_demo,
                lat_demo,
                p_level_demo,
                midpoint=self.flag_midpoint,
            )
            cfg_ef = post_conf["global_energy_fixer"]
            self.N_seconds = int(cfg_ef["lead_time_periods"]) * 3600

            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)

        else:
            # the actual setup for model runs
            cfg_ef = post_conf["global_energy_fixer"]
            ds_physics = get_forward_data(cfg_ef["save_loc_physics"])

            lon_lat_level_names = cfg_ef.get(
                "lon_lat_level_name", post_conf["global_mass_fixer"]["lon_lat_level_name"]
            )
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = cfg_ef.get(
                "midpoint", post_conf["global_mass_fixer"]["midpoint"]
            )
            grid_type = cfg_ef.get(
                "grid_type", post_conf["global_mass_fixer"]["grid_type"]
            )

            if grid_type == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)

                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, midpoint=self.flag_midpoint)

            self.N_seconds = int(cfg_ef["lead_time_periods"]) * 3600

            varname_gph = cfg_ef["surface_geopotential_name"]
            self.GPH_surf = torch.from_numpy(ds_physics[varname_gph[0]].values).float()

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        cfg_ef = post_conf["global_energy_fixer"]
        self.T_ind_start = int(cfg_ef["T_inds"][0])
        self.T_ind_end = int(cfg_ef["T_inds"][-1]) + 1

        self.q_ind_start = int(cfg_ef["q_inds"][0])
        self.q_ind_end = int(cfg_ef["q_inds"][-1]) + 1

        self.U_ind_start = int(cfg_ef["U_inds"][0])
        self.U_ind_end = int(cfg_ef["U_inds"][-1]) + 1

        self.V_ind_start = int(cfg_ef["V_inds"][0])
        self.V_ind_end = int(cfg_ef["V_inds"][-1]) + 1

        self.TOA_solar_ind = int(cfg_ef["TOA_rad_inds"][0])
        self.TOA_OLR_ind = int(cfg_ef["TOA_rad_inds"][1])

        self.surf_solar_ind = int(cfg_ef["surf_rad_inds"][0])
        self.surf_LR_ind = int(cfg_ef["surf_rad_inds"][1])

        self.surf_SH_ind = int(cfg_ef["surf_flux_inds"][0])
        self.surf_LH_ind = int(cfg_ef["surf_flux_inds"][1])

        if self.flag_sigma_level:
            self.sp_ind = int(cfg_ef["sp_inds"])
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if cfg_ef.get("denorm", False):
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x["x"]
        y_pred = x["y_pred"]

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        GPH_surf = self.GPH_surf.to(y_pred.device)
        N_vars = y_pred.shape[1]

        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input)
            y_pred = self.state_trans.inverse_transform(y_pred)

        T_input = x_input[:, self.T_ind_start : self.T_ind_end, -1, ...]
        q_input = x_input[:, self.q_ind_start : self.q_ind_end, -1, ...]
        U_input = x_input[:, self.U_ind_start : self.U_ind_end, -1, ...]
        V_input = x_input[:, self.V_ind_start : self.V_ind_end, -1, ...]

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        T_pred = y_pred[:, self.T_ind_start : self.T_ind_end, 0, ...]
        q_pred = y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...]
        U_pred = y_pred[:, self.U_ind_start : self.U_ind_end, 0, ...]
        V_pred = y_pred[:, self.V_ind_start : self.V_ind_end, 0, ...]

        TOA_solar_pred = y_pred[:, self.TOA_solar_ind, 0, ...]
        TOA_OLR_pred = y_pred[:, self.TOA_OLR_ind, 0, ...]

        surf_solar_pred = y_pred[:, self.surf_solar_ind, 0, ...]
        surf_LR_pred = y_pred[:, self.surf_LR_ind, 0, ...]
        surf_SH_pred = y_pred[:, self.surf_SH_ind, 0, ...]
        surf_LH_pred = y_pred[:, self.surf_LH_ind, 0, ...]

        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...]
            sp_pred = y_pred[:, self.sp_ind, 0, ...]

        # ------------------------------------------------------------------------------ #
        # Latent heat, potential energy, kinetic energy

        # heat capacity on constant pressure
        CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
        CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR

        # kinetic energy
        ken_t0 = 0.5 * (U_input**2 + V_input**2)
        ken_t1 = 0.5 * (U_pred**2 + V_pred**2)

        # packing latent heat + potential energy + kinetic energy
        E_qgk_t0 = LH_WATER * q_input + GPH_surf + ken_t0
        E_qgk_t1 = LH_WATER * q_pred + GPH_surf + ken_t1

        # ------------------------------------------------------------------------------ #
        # energy source and sinks

        # TOA energy flux
        R_T = (TOA_solar_pred + TOA_OLR_pred) / self.N_seconds
        R_T_sum = self.core_compute.weighted_sum(R_T, axis=(-2, -1))

        # surface net energy flux
        F_S = (surf_solar_pred + surf_LR_pred + surf_SH_pred + surf_LH_pred) / self.N_seconds
        F_S_sum = self.core_compute.weighted_sum(F_S, axis=(-2, -1))

        # ------------------------------------------------------------------------------ #
        # thermal energy correction

        # total energy per level
        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        # column integrated total energy
        if self.flag_sigma_level:
            TE_t0 = self.core_compute.integral(E_level_t0, sp_input) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1, sp_pred) / GRAVITY
        else:
            TE_t0 = self.core_compute.integral(E_level_t0) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1) / GRAVITY

        # dTE_dt = (TE_t1 - TE_t0) / self.N_seconds

        global_TE_t0 = self.core_compute.weighted_sum(TE_t0, axis=(-2, -1))
        global_TE_t1 = self.core_compute.weighted_sum(TE_t1, axis=(-2, -1))

        # total energy correction ratio
        E_correct_ratio = (self.N_seconds * (R_T_sum - F_S_sum) + global_TE_t0) / global_TE_t1
        # E_correct_ratio = torch.clamp(E_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch, 1, 1, 1, 1)
        E_correct_ratio = E_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # apply total energy correction
        E_t1_correct = E_level_t1 * E_correct_ratio

        # let thermal energy carry the corrected total energy amount
        T_pred = (E_t1_correct - E_qgk_t1) / CP_t1

        # ===================================================================== #
        # return fixed q and precip back to y_pred

        # expand fixed vars to (batch level, time, lat, lon)
        T_pred = T_pred.unsqueeze(2)

        y_pred = concat_fix(y_pred, T_pred, self.T_ind_start, self.T_ind_end, N_vars)

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalEnergyFixerUpDown(nn.Module):
    """
    Global energy conservation fixer using explicit up/down flux decomposition.

    Identical correction logic to ``GlobalEnergyFixer`` but uses separate downwelling
    and upwelling flux indices rather than pre-computed net fluxes.  The net TOA and
    surface imbalances are formed as:

    .. code-block:: text

        R_T  = (DSWRFtoa  - USWRFtoa  - ULWRFtoa) / N_seconds
        F_S  = (FSDS_J    - FSUS      + FLDS_J    - FLUS - SHF - LHF) / N_seconds

    where ``*_J`` variables are in J/m² (energy over the timestep) and ``SHF``/``LHF``
    are positive-upward surface turbulent heat fluxes also in J/m².

    Args:
        post_conf (dict): config dictionary.  The sub-key ``global_energy_fixer_updown``
            must be present and contain all specs listed below.

    Config keys (under ``global_energy_fixer_updown``):
        - ``activate`` / ``activate_outside_model`` / ``simple_demo``
        - ``midpoint``, ``denorm``, ``surface_geopotential_name``
        - ``T_inds``, ``q_inds``, ``U_inds``, ``V_inds``
        - ``sp_inds``  (required when ``grid_type == 'sigma'``)
        - ``TOA_forcing_solar_ind`` — SOLIN channel index in the *input* (x) tensor (W/m²);
          the fixer multiplies by ``N_seconds`` internally to convert to J/m²
        - ``TOA_up_solar_ind``   — FSUTOA_J index in y_pred (J/m²)
        - ``TOA_up_OLR_ind``     — FLUT_J index in y_pred (J/m²)
        - ``surf_down_solar_ind`` — FSDS_J index in y_pred
        - ``surf_up_solar_ind``  — FSUS index in y_pred
        - ``surf_down_LW_ind``   — FLDS_J index in y_pred
        - ``surf_up_LW_ind``     — FLUS index in y_pred
        - ``surf_SH_ind``        — SHF index in y_pred (same sign convention as FLNS/SHFLX in zarr:
          negative = upward, i.e. stored as ``-(positive_upward × DT)``)
        - ``surf_LH_ind``        — LHF index in y_pred (same sign convention as SHF)
    """

    def __init__(self, post_conf):
        super().__init__()

        cfg = post_conf["global_energy_fixer_updown"]

        # ------------------------------------------------------------------ #
        # Grid / physics setup — reuse the same logic as GlobalEnergyFixer
        if cfg["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340])
            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)
            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = cfg["midpoint"]
            self.core_compute = physics_pressure_level(lon_demo, lat_demo, p_level_demo, midpoint=self.flag_midpoint)
            self.N_seconds = int(cfg["lead_time_periods"]) * 3600
            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)
        else:
            ds_physics = get_forward_data(cfg["save_loc_physics"])
            lon_lat_level_names = cfg["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            self.flag_midpoint = cfg["midpoint"]

            if cfg["grid_type"] == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).float()
                self.N_levels = len(self.coef_a)
                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1
                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                self.N_levels = len(p_level)
                self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, midpoint=self.flag_midpoint)

            self.N_seconds = int(cfg["lead_time_periods"]) * 3600

            varname_gph = cfg["surface_geopotential_name"]
            self.GPH_surf = torch.from_numpy(ds_physics[varname_gph[0]].values).float()

        # ------------------------------------------------------------------ #
        # Variable indices — atmosphere state
        self.T_ind_start = int(cfg["T_inds"][0])
        self.T_ind_end = int(cfg["T_inds"][-1]) + 1
        self.q_ind_start = int(cfg["q_inds"][0])
        self.q_ind_end = int(cfg["q_inds"][-1]) + 1
        self.U_ind_start = int(cfg["U_inds"][0])
        self.U_ind_end = int(cfg["U_inds"][-1]) + 1
        self.V_ind_start = int(cfg["V_inds"][0])
        self.V_ind_end = int(cfg["V_inds"][-1]) + 1

        if self.flag_sigma_level:
            self.sp_ind = int(cfg["sp_inds"])

        # ------------------------------------------------------------------ #
        # Variable indices — up/down fluxes
        # SOLIN is a forcing variable; read from x (input tensor) rather than y_pred
        self.TOA_forcing_solar_ind = int(cfg["TOA_forcing_solar_ind"])
        self.TOA_up_solar_ind = int(cfg["TOA_up_solar_ind"])
        self.TOA_up_OLR_ind = int(cfg["TOA_up_OLR_ind"])

        self.surf_down_solar_ind = int(cfg["surf_down_solar_ind"])
        self.surf_up_solar_ind = int(cfg["surf_up_solar_ind"])
        self.surf_down_LW_ind = int(cfg["surf_down_LW_ind"])
        self.surf_up_LW_ind = int(cfg["surf_up_LW_ind"])
        self.surf_SH_ind = int(cfg["surf_SH_ind"])
        self.surf_LH_ind = int(cfg["surf_LH_ind"])

        # How often to log the energy-budget diagnostic (forward passes). Set to 1
        # in a rollout to trace every step; the default keeps long-run logs quiet.
        self.log_every = int(cfg.get("log_every", 50))

        # ------------------------------------------------------------------ #
        # Optional denorm: load per-variable mean/std directly from NC files.
        # We do selective channel-wise denorm in forward() rather than full
        # tensor denorm, because statistics can be scalar, level-varying (1-D),
        # or spatially-varying (H×W) depending on the variable.
        self.flag_denorm = bool(cfg.get("denorm", False))
        if self.flag_denorm:
            mean_ds = xr.open_dataset(cfg["mean_path"]).load()
            std_ds = xr.open_dataset(cfg["std_path"]).load()

            def _buf(varname, expand_dims=None):
                """Return (mean, std) as float tensors; expand_dims = list of axes to unsqueeze."""
                m = torch.from_numpy(np.array(mean_ds[varname].values)).float()
                s = torch.from_numpy(np.array(std_ds[varname].values)).float()
                if expand_dims:
                    for d in expand_dims:
                        m = m.unsqueeze(d)
                        s = s.unsqueeze(d)
                return m, s

            # 3-D prognostic variables: stats shape (n_levels,) → (1, n_levels, 1, 1)
            T_m, T_s = _buf("T", expand_dims=[0, -1, -1])    # (1, 32, 1, 1)
            q_m, q_s = _buf("Qtot", expand_dims=[0, -1, -1])
            U_m, U_s = _buf("U", expand_dims=[0, -1, -1])
            V_m, V_s = _buf("V", expand_dims=[0, -1, -1])

            # 2-D spatial variables: stats shape (H, W) → (1, H, W)
            # One unsqueeze so (H,W) broadcasts against (B, H, W) without shifting dims
            PS_m, PS_s = _buf("PS", expand_dims=[0])           # (1, H, W)

            # 2-D scalar variables: scalar → ()
            def _scalar_buf(vname):
                return _buf(vname)   # already scalar shape ()

            SOLIN_m,  SOLIN_s  = _scalar_buf("SOLIN")
            FSUTOA_m, FSUTOA_s = _scalar_buf("FSUTOA")
            FLUT_m,   FLUT_s   = _scalar_buf("FLUT")
            FSDS_J_m, FSDS_J_s = _scalar_buf("FSDS_J")
            FLDS_J_m, FLDS_J_s = _scalar_buf("FLDS_J")
            FSUS_m,   FSUS_s   = _scalar_buf("FSUS")
            FLUS_m,   FLUS_s   = _scalar_buf("FLUS")
            SHFLX_m,  SHFLX_s  = _scalar_buf("SHFLX")
            LHFLX_m,  LHFLX_s  = _scalar_buf("LHFLX")

            # Store as plain attributes (NOT register_buffer) so they are:
            #   (a) NOT included in state_dict → EMA won't try to track them
            #   (b) NOT saved in checkpoints (they're reloaded from NC at init time)
            # Device placement is handled lazily in _denorm/_renorm via .to(device).
            self.T_mean = T_m;      self.T_std = T_s
            self.q_mean = q_m;      self.q_std = q_s
            self.U_mean = U_m;      self.U_std = U_s
            self.V_mean = V_m;      self.V_std = V_s
            self.PS_mean = PS_m;    self.PS_std = PS_s
            self.SOLIN_mean = SOLIN_m;   self.SOLIN_std = SOLIN_s
            self.FSUTOA_mean = FSUTOA_m; self.FSUTOA_std = FSUTOA_s
            self.FLUT_mean = FLUT_m;     self.FLUT_std = FLUT_s
            self.FSDS_J_mean = FSDS_J_m; self.FSDS_J_std = FSDS_J_s
            self.FLDS_J_mean = FLDS_J_m; self.FLDS_J_std = FLDS_J_s
            self.FSUS_mean = FSUS_m;     self.FSUS_std = FSUS_s
            self.FLUS_mean = FLUS_m;     self.FLUS_std = FLUS_s
            self.SHFLX_mean = SHFLX_m;   self.SHFLX_std = SHFLX_s
            self.LHFLX_mean = LHFLX_m;   self.LHFLX_std = LHFLX_s

    def _denorm(self, vals, mean, std):
        """Inverse-normalize in float32 (physics integrals need full precision)."""
        m = mean.to(device=vals.device, dtype=torch.float32)
        s = std.to(device=vals.device, dtype=torch.float32)
        return vals.float() * s + m

    def _renorm(self, vals, mean, std):
        """Re-normalize, keeping float32 output."""
        m = mean.to(device=vals.device, dtype=torch.float32)
        s = std.to(device=vals.device, dtype=torch.float32)
        return (vals.float() - m) / s

    def forward(self, x):
        x_input = x["x"]
        y_pred = x["y_pred"]
        orig_dtype = y_pred.dtype          # remember for output cast-back
        x_input = x_input.detach().to(y_pred.device)

        GPH_surf = self.GPH_surf.to(device=y_pred.device, dtype=torch.float32)
        N_vars = y_pred.shape[1]

        # ------------------------------------------------------------------ #
        # Atmosphere state at t0 and t1 — denorm if requested
        if self.flag_denorm:
            T_input = self._denorm(x_input[:, self.T_ind_start:self.T_ind_end, -1, ...], self.T_mean, self.T_std)
            q_input = self._denorm(x_input[:, self.q_ind_start:self.q_ind_end, -1, ...], self.q_mean, self.q_std)
            U_input = self._denorm(x_input[:, self.U_ind_start:self.U_ind_end, -1, ...], self.U_mean, self.U_std)
            V_input = self._denorm(x_input[:, self.V_ind_start:self.V_ind_end, -1, ...], self.V_mean, self.V_std)

            T_pred = self._denorm(y_pred[:, self.T_ind_start:self.T_ind_end, 0, ...], self.T_mean, self.T_std)
            q_pred = self._denorm(y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...], self.q_mean, self.q_std)
            U_pred = self._denorm(y_pred[:, self.U_ind_start:self.U_ind_end, 0, ...], self.U_mean, self.U_std)
            V_pred = self._denorm(y_pred[:, self.V_ind_start:self.V_ind_end, 0, ...], self.V_mean, self.V_std)

            if self.flag_sigma_level:
                sp_input = self._denorm(x_input[:, self.sp_ind, -1, ...], self.PS_mean, self.PS_std)
                sp_pred  = self._denorm(y_pred[:, self.sp_ind, 0, ...],   self.PS_mean, self.PS_std)
        else:
            T_input = x_input[:, self.T_ind_start:self.T_ind_end, -1, ...]
            q_input = x_input[:, self.q_ind_start:self.q_ind_end, -1, ...]
            U_input = x_input[:, self.U_ind_start:self.U_ind_end, -1, ...]
            V_input = x_input[:, self.V_ind_start:self.V_ind_end, -1, ...]

            T_pred = y_pred[:, self.T_ind_start:self.T_ind_end, 0, ...]
            q_pred = y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...]
            U_pred = y_pred[:, self.U_ind_start:self.U_ind_end, 0, ...]
            V_pred = y_pred[:, self.V_ind_start:self.V_ind_end, 0, ...]

            if self.flag_sigma_level:
                sp_input = x_input[:, self.sp_ind, -1, ...]
                sp_pred  = y_pred[:, self.sp_ind, 0, ...]

        # ------------------------------------------------------------------ #
        # Latent heat, potential energy, kinetic energy
        CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
        CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR

        ken_t0 = 0.5 * (U_input**2 + V_input**2)
        ken_t1 = 0.5 * (U_pred**2 + V_pred**2)

        E_qgk_t0 = LH_WATER * q_input + GPH_surf + ken_t0
        E_qgk_t1 = LH_WATER * q_pred + GPH_surf + ken_t1

        # ------------------------------------------------------------------ #
        # TOA net flux: down_SW - up_SW - up_LW  (positive = energy in)
        # SOLIN from x (forcing, W/m²); FSUTOA, FLUT from y_pred (W/m²)
        if self.flag_denorm:
            TOA_down_solar = self._denorm(x_input[:, self.TOA_forcing_solar_ind, -1, ...], self.SOLIN_mean,  self.SOLIN_std)  * self.N_seconds
            TOA_up_solar   = self._denorm(y_pred[:, self.TOA_up_solar_ind, 0, ...],        self.FSUTOA_mean, self.FSUTOA_std) * self.N_seconds
            TOA_up_OLR     = self._denorm(y_pred[:, self.TOA_up_OLR_ind, 0, ...],          self.FLUT_mean,   self.FLUT_std)   * self.N_seconds
        else:
            TOA_down_solar = x_input[:, self.TOA_forcing_solar_ind, -1, ...] * self.N_seconds
            TOA_up_solar   = y_pred[:, self.TOA_up_solar_ind, 0, ...] * self.N_seconds
            TOA_up_OLR     = y_pred[:, self.TOA_up_OLR_ind, 0, ...] * self.N_seconds

        R_T = (TOA_down_solar - TOA_up_solar - TOA_up_OLR) / self.N_seconds
        R_T_sum = self.core_compute.weighted_sum(R_T, axis=(-2, -1))

        # Surface net flux: (down_SW - up_SW) + (down_LW - up_LW) + SHF + LHF
        # zarr convention: SHFLX/LHFLX negative = upward; + sign adds upward heat to atmosphere
        if self.flag_denorm:
            surf_down_solar = self._denorm(y_pred[:, self.surf_down_solar_ind, 0, ...], self.FSDS_J_mean, self.FSDS_J_std)
            surf_up_solar   = self._denorm(y_pred[:, self.surf_up_solar_ind, 0, ...],   self.FSUS_mean,   self.FSUS_std)
            surf_down_LW    = self._denorm(y_pred[:, self.surf_down_LW_ind, 0, ...],    self.FLDS_J_mean, self.FLDS_J_std)
            surf_up_LW      = self._denorm(y_pred[:, self.surf_up_LW_ind, 0, ...],      self.FLUS_mean,   self.FLUS_std)
            surf_SH         = self._denorm(y_pred[:, self.surf_SH_ind, 0, ...],         self.SHFLX_mean,  self.SHFLX_std)
            surf_LH         = self._denorm(y_pred[:, self.surf_LH_ind, 0, ...],         self.LHFLX_mean,  self.LHFLX_std)
        else:
            surf_down_solar = y_pred[:, self.surf_down_solar_ind, 0, ...]
            surf_up_solar   = y_pred[:, self.surf_up_solar_ind, 0, ...]
            surf_down_LW    = y_pred[:, self.surf_down_LW_ind, 0, ...]
            surf_up_LW      = y_pred[:, self.surf_up_LW_ind, 0, ...]
            surf_SH         = y_pred[:, self.surf_SH_ind, 0, ...]
            surf_LH         = y_pred[:, self.surf_LH_ind, 0, ...]

        F_S = (surf_down_solar - surf_up_solar + surf_down_LW - surf_up_LW + surf_SH + surf_LH) / self.N_seconds
        F_S_sum = self.core_compute.weighted_sum(F_S, axis=(-2, -1))

        # ------------------------------------------------------------------ #
        # Column total energy and correction ratio
        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        if self.flag_sigma_level:
            TE_t0 = self.core_compute.integral(E_level_t0, sp_input) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1, sp_pred) / GRAVITY
        else:
            TE_t0 = self.core_compute.integral(E_level_t0) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1) / GRAVITY

        global_TE_t0 = self.core_compute.weighted_sum(TE_t0, axis=(-2, -1))
        global_TE_t1 = self.core_compute.weighted_sum(TE_t1, axis=(-2, -1))

        flux_correction = self.N_seconds * (R_T_sum - F_S_sum)   # J/m²
        expected_TE_t1 = global_TE_t0 + flux_correction
        E_correct_ratio = expected_TE_t1 / global_TE_t1

        self._fixer_calls = getattr(self, "_fixer_calls", 0) + 1
        log_now = (self._fixer_calls % self.log_every == 1) or self.log_every == 1

        # Guard: the fixer is only meaningful on PHYSICAL units. If it is ever fed
        # normalized values (denorm misconfigured, or mean/std paths wrong), every
        # budget term below is garbage and the rollout will diverge. Fail loudly here
        # rather than mysteriously hundreds of steps into a run.
        if log_now or self._fixer_calls == 1:
            t_mean = T_input.detach().float().mean().item()
            if not (150.0 < t_mean < 350.0):
                logger.error(
                    "EnergyFixer: T_input global mean = %.3f K is not physical (expected 150-350 K). "
                    "The fixer is being fed NORMALIZED data — check denorm / mean_path / std_path. "
                    "Energy corrections are meaningless until this is fixed.",
                    t_mean,
                )

        E_correct_ratio_b = E_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        E_t1_correct = E_level_t1 * E_correct_ratio_b
        T_pred_corrected = (E_t1_correct - E_qgk_t1) / CP_t1

        # ---- Diagnostics: the RESIDUAL CORRECTION MAGNITUDE -------------------
        # `residual` is the gap between the model's own energy tendency and what its
        # predicted fluxes claim; the fixer forces that gap onto T every step. For
        # reference, ERA5-scaled truth closes this budget to ~0.4 W/m^2. Reported in
        # W/m^2 and K/step because the raw total energy (~1.3e24 J) is far too large
        # for a fatal drift to be visible in it.
        if log_now:
            with torch.no_grad():
                area = self.core_compute.area.to(global_TE_t0.device).sum()
                flux_tendency = ((R_T_sum - F_S_sum) / area).float()            # W/m^2
                model_tendency = (
                    (global_TE_t1 - global_TE_t0) / self.N_seconds / area
                ).float()                                                        # W/m^2
                residual = flux_tendency - model_tendency                        # W/m^2
                dT = (T_pred_corrected - T_pred).detach().float()                # K
                r = E_correct_ratio.detach().float()
                logger.info(
                    "EnergyFixer step %d | residual=%+.3f W/m2 "
                    "(model dTE/dt=%+.3f, fluxes say R_T-F_S=%+.3f) | "
                    "correction dT mean=%+.4f K  |dT|max=%.4f K | "
                    "ratio mean=%.6f min=%.6f max=%.6f",
                    self._fixer_calls,
                    residual.mean().item(),
                    model_tendency.mean().item(),
                    flux_tendency.mean().item(),
                    dT.mean().item(),
                    dT.abs().max().item(),
                    r.mean().item(), r.min().item(), r.max().item(),
                )

        T_pred = T_pred_corrected

        # ------------------------------------------------------------------ #
        # Write corrected T back to y_pred (re-normalize if denorm was applied)
        if self.flag_denorm:
            T_pred = self._renorm(T_pred, self.T_mean, self.T_std)
        # Cast back to original dtype (e.g. bfloat16) before splicing into y_pred
        T_pred = T_pred.to(dtype=orig_dtype).unsqueeze(2)
        y_pred = concat_fix(y_pred, T_pred, self.T_ind_start, self.T_ind_end, N_vars)

        x["y_pred"] = y_pred
        return x


def concat_fix(y_pred, q_pred_correct, q_ind_start, q_ind_end, N_vars):
    """
    this function use torch.concat to replace a specific subset of variable channels in `y_pred`.

    Given `q_pred = y_pred[:, ind_start:ind_end, ...]`, and `q_pred_correct` this function
    does: `y_pred[:, ind_start:ind_end, ...] = q_pred_correct`, but without using in-place
    modifications, so the graph of y_pred is maintained. It also handles
    `q_ind_start == q_ind_end cases`.

    All input tensors must have 5 dims of `batch, level-or-var, time, lat, lon`

    Args:
        y_pred (torch.Tensor): Original y_pred tensor of shape (batch, var, time, lat, lon).
        q_pred_correct (torch.Tensor): Corrected q_pred tensor.
        q_ind_start (int): Index where q_pred starts in y_pred.
        q_ind_end (int): Index where q_pred ends in y_pred.
        N_vars (int): Total number of variables in y_pred (i.e., y_pred.shape[1]).

    Returns:
        torch.Tensor: Concatenated y_pred with corrected q_pred.
    """
    # define a list that collects tensors
    var_list = []

    # vars before q_pred
    if q_ind_start > 0:
        var_list.append(y_pred[:, :q_ind_start, ...])

    # q_pred
    var_list.append(q_pred_correct)

    # vars after q_pred
    if q_ind_end < N_vars - 1:
        if q_ind_start == q_ind_end:
            var_list.append(y_pred[:, q_ind_end + 1 :, ...])
        else:
            var_list.append(y_pred[:, q_ind_end:, ...])

    return torch.cat(var_list, dim=1)
