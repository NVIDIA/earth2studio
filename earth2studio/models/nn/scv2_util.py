import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from physicsnemo.experimental.models.dit.dit import DiT as PNM_DiT

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model=None,  # instance of the model to be used
        return_logvar=False,
        logvar_channels=128,
        output_channels=30,
        dropout: bool = False,
        sigma_max_dropout: float = 1000.0,
        sigma_min_dropout: float = 0.002,
        dropout_function_type: str = "sigmoid",
        p_max: float = 0.9,
        p_min: float = 0.1,
        x_offset: float = 15.0,
        slope: float = 6.0,
    ):
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model
        self.return_logvar = return_logvar
        if self.return_logvar:
            self.logvar_fourier = MPFourier(logvar_channels)
            self.logvar_linear = MPConv(logvar_channels, output_channels, kernel=[])

        if dropout:
            self.noise_dependent_dropout = dropout
            self.sigma_max_dropout = torch.tensor(sigma_max_dropout)
            self.sigma_min_dropout = torch.tensor(sigma_min_dropout)
            self.dropout_function_type = dropout_function_type
            self.p_max = p_max
            self.p_min = p_min
            self.x_offset = x_offset
            self.slope = slope
            print(f"sigma_max_dropout: {self.sigma_max_dropout}")
            print(f"sigma_min_dropout: {self.sigma_min_dropout}")
            print(f"dropout_function_type: {self.dropout_function_type}")
            print(f"p_max: {self.p_max}")
            print(f"p_min: {self.p_min}")
            print(f"x_offset: {self.x_offset}")
            print(f"slope: {self.slope}")
        else:
            self.noise_dependent_dropout = False

    def forward(
        self,
        x,
        sigma,
        condition,
        class_labels=None,
        return_logvar=False,
        force_fp32=False,
        training=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        """
        p_dropout is the dropout probability for the model.
        class DropoutConfig:
        dropout: bool = False
        sigma_max_dropout: float = 200.0
        sigma_min_dropout: float = 0.002
        dropout_function_type: str = "sigmoid"
        p_max: float = 0.9
        p_min: float = 0.1
        x_offset: float = 15.0
        slope: float = 6.0
        """

        p_dropout = 0
        if self.noise_dependent_dropout:
            if self.dropout_function_type == "sigmoid":
                x_offset = torch.tensor(
                    self.x_offset
                )  # this is the point where the dropout probability is 0.5
                log_sigma = torch.log(sigma)
                log_offset = torch.log(x_offset)
                sigmoid = 1 / (
                    1 + torch.exp(-self.slope * (log_sigma - log_offset))
                )  # slope is the steepness of the sigmoid function for S curve
                p_dropout = (
                    self.p_min + (self.p_max - self.p_min) * sigmoid
                )  # p_min is the minimum dropout probability, p_max is the maximum dropout probability
            else:
                log_sigma_range = torch.log(self.sigma_max_dropout) - torch.log(
                    self.sigma_min_dropout
                )
                # Clamp sigma to [0, max_sigma] to ensure r is in [0, 1]
                r = (
                    torch.clamp(
                        torch.log(sigma) - torch.log(self.sigma_min_dropout),
                        max=log_sigma_range,
                    )
                    / log_sigma_range
                )
                p_dropout = (self.p_max - self.p_min) * r + self.p_min
        else:
            p_dropout = None

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)
        # now we have added the p_dropout probability to the model.
        F_x = self.model(
            (arg).to(dtype),
            c_noise.flatten(),
            p_dropout=p_dropout,
            training=training,
            **model_kwargs,
        )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        if return_logvar:
            logvar = (
                self.logvar_linear(self.logvar_fourier(c_noise.flatten()))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            return D_x, logvar  # u(sigma) in Equation 21

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class DropInDiT(nn.Module):
    """
    Wrapper that exposes the old DiT API while delegating to PhysicsNeMo DiT.

    Forward signature matches `models.diffusion_transformer.DiT.forward`:
        forward(x, time_step_cond=None, label_cond=None, points=None, p_dropout=None, training=False)
    """

    def __init__(
        self,
        pnm: PNM_DiT,
    ):
        super().__init__()
        self.pnm = pnm

        # Cache tokenizer geometry for NAT2D latent_hw
        self._input_size: Tuple[int, int] = tuple(int(x) for x in pnm.input_size)
        self._patch_size: Tuple[int, int] = tuple(int(x) for x in pnm.patch_size)

    @torch.no_grad()
    def _compute_latent_hw(self, x: torch.Tensor) -> Tuple[int, int]:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ph, pw = self._patch_size
        return h // ph, w // pw

    def forward(
        self,
        x: torch.Tensor,
        time_step_cond: Optional[torch.Tensor] = None,
        label_cond: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        p_dropout: Optional[float | torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        # time_step_cond required by PNM; default to zeros if None
        if time_step_cond is None:
            time_step_cond = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # PNM DiT uses a combined condition vector c = t (+ optional extra condition embedding)
        condition = None

        # Always provide NAT latent_hw
        latent_hw = self._compute_latent_hw(x)
        attn_kwargs: Optional[Dict[str, Any]] = {"latent_hw": latent_hw}

        # Note: points / cross-attention are not supported in PhysicsNeMo DiT so we ignore them
        out = self.pnm(
            x=x,
            t=time_step_cond,
            condition=condition,
            p_dropout=p_dropout,
            attn_kwargs=attn_kwargs,
        )
        return out

def edm_sampler(
    net,
    latents,
    condition=None,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=800,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    progress_bar=None,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Select the active network for the current step
        active_net = net

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = active_net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        nans = np.sum(np.isnan(x_hat.cpu().numpy()))
        if nans > 0:
            print("NANs", nans, "at step", i, x_hat.shape)
        denoised = active_net(
            x_hat, t_hat, class_labels=class_labels, condition=condition
        ).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # Select the active network for the next step
            active_net_prime = net

            denoised = active_net_prime(
                x_next, t_next, class_labels=class_labels, condition=condition
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if progress_bar:
            progress_bar.update()

    return x_next
