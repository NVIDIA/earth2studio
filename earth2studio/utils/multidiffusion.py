import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from tqdm.auto import tqdm

class MultiDiffusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        net: torch.nn.Module,
        img_lr: Tensor,
        regression_output: Tensor,
        class_labels: Optional[Tensor] = None,
        randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
        windows:  Optional[Tensor] = None,
        lead_time_label: Optional[Tensor] = None,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ) -> Tensor:
        """
        
        Args:
            net (torch.nn.Module): the diffusion model
            regression_output (Tensor): output from regression model (B, C_cond, H, W)。
            randn_like (Callable): gaussian sampler
            windows : All windows
            stride (int): the stride between windows

        Returns:
            Tensor: (B, C_out, H, W)
        """
        
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        batch_size, _, height, width = regression_output.shape
        x_lr = torch.cat((regression_output,img_lr), dim=1)
        latents = randn_like(regression_output)
        
        step_indices = torch.arange(num_steps, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        views = windows
        
        value = torch.zeros_like(latents)
        count = torch.zeros_like(latents)

        optional_args = {}
        if lead_time_label is not None:
            optional_args["lead_time_label"] = lead_time_label

        x_next = latents * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(tqdm(zip(t_steps[:-1], t_steps[1:]), total=num_steps)):
            x_cur = x_next.clone()
            
            value.zero_()
            count.zero_()
            #print(f"x_cur:{x_cur.shape}")
            for view in views:
                h_start, h_end, w_start, w_end = int(view[0]),int(view[1]),int(view[2]),int(view[3])
                x_cur_view = x_cur[:, :, h_start:h_end, w_start:w_end]
                x_lr_view = x_lr[:, :, h_start:h_end, w_start:w_end]
                
                #(Churning)
                gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
                t_hat = net.round_sigma(t_cur + gamma * t_cur)
                x_hat_view = x_cur_view + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur_view)
                #(Euler step part 1)
                denoised_view = net(
                    x_hat_view,
                    x_lr_view,
                    t_hat,
                    class_labels,
                    **optional_args,
                )
                
                d_cur_view = (x_hat_view - denoised_view) / t_hat
                
                x_next_view_first_order = x_hat_view + (t_next - t_hat) * d_cur_view

                if i < num_steps - 1:
                    denoised_prime_view = net(
                        x_next_view_first_order,
                        x_lr_view,
                        t_next,
                        class_labels,
                        **optional_args,
                    )
                    d_prime_view = (x_next_view_first_order - denoised_prime_view) / t_next
                    x_next_view = x_hat_view + (t_next - t_hat) * (0.5 * d_cur_view + 0.5 * d_prime_view)
                else:
                    x_next_view = x_next_view_first_order
                
                value[:, :, h_start:h_end, w_start:w_end] += x_next_view
                count[:, :, h_start:h_end, w_start:w_end] += 1
                #print(f"One window finished.")

            x_next = torch.where(count > 0, value / count, value)
        
        return x_next
