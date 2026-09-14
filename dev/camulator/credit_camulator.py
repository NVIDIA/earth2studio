import torch
import logging
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from torch import nn as _nn
BaseModel = _nn.Module
from credit_boundary_padding import TensorPadding

logger = logging.getLogger(__name__)

# helpers


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def apply_spectral_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            # skip any conv whose name contains "sharp"
            if "sharp" in name:
                continue
            nn.utils.spectral_norm(module)


# cube embedding


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, T, C, Lat, Lon = x.shape
        x = self.proj(x)

        # ----------------------------------- #
        # Layer norm on T*lat*lon
        x = x.reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)

        return x.squeeze(2)


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        # self.upsample = nn.functional.interpolate(scale_factor=2, mode="bilinear", align_corners=False, antialias=True)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.sharp = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1, bias=True)
        nn.init.zeros_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        self.output_channels = out_chans

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False, antialias=False)
        x = self.conv(x)
        x = x + self.sharp(x)
        shortcut = x

        x = self.b(x)

        return x + shortcut


class UpBlockPS(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups, scale=2, num_residuals=2):
        super().__init__()
        # sub-pixel conv at low res
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale)
        # sharpening branch (identity init)
        self.sharp = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.xavier_normal_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        # residual stack
        blk = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
            ]
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.ps(self.conv(x))  # upsample+conv at low res
        x = x + self.sharp(x)  # sharpen residual
        sc = x
        x = self.b(x)
        return x + sc


# cross embed layer


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


# dynamic positional bias


class DynamicPositionBias(nn.Module):
    def __init__(self, dim):
        super(DynamicPositionBias, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            Rearrange("... () -> ..."),
        )

    def forward(self, x):
        return self.layers(x)


# transformer classes


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0):
        super().__init__()
        assert attn_type in {
            "short",
            "long",
        }, "attention type must be one of local or distant"
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        # positions

        self.dpb = DynamicPositionBias(dim // 4)

        # calculate and store indices for retrieving bias

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        *_, height, width, heads, wsz, device = (
            *x.shape,
            self.heads,
            self.window_size,
            x.device,
        )

        # prenorm

        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == "short":
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz, s2=wsz)
        elif self.attn_type == "long":
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz, l2=wsz)

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add dynamic positional bias

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rearrange(rel_pos, "c i j -> (i j) c")
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz, y=wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == "short":
            out = rearrange(
                out,
                "(b h w) d s1 s2 -> b d (h s1) (w s2)",
                h=height // wsz,
                w=width // wsz,
            )
        elif self.attn_type == "long":
            out = rearrange(
                out,
                "(b h w) d l1 l2 -> b d (l1 h) (l2 w)",
                h=height // wsz,
                w=width // wsz,
            )

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth=4,
        dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            attn_type="short",
                            window_size=local_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                        Attention(
                            dim,
                            attn_type="long",
                            window_size=global_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


# classes


class Camulator(BaseModel):
    def __init__(
        self,
        image_height: int = 640,
        patch_height: int = 1,
        image_width: int = 1280,
        patch_width: int = 1,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size: tuple = (5, 5, 2, 1),
        local_window_size: int = 10,
        cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides: tuple = (4, 2, 2, 2),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_spectral_norm: bool = True,
        interp: bool = True,
        padding_conf: dict = None,
        post_conf: dict = None,
        **kwargs,
    ):
        """
        CrossFormer is the base architecture for the WXFormer model. It uses convolutions and long and short distance
        attention layers in the encoder layer and then uses strided transpose convolution blocks for the decoder
        layer.

        Args:
            image_height (int): number of grid cells in the south-north direction.
            patch_height (int): number of grid cells within each patch in the south-north direction.
            image_width (int): number of grid cells in the west-east direction.
            patch_width (int): number of grid cells within each patch in the west-east direction.
            frames (int): number of time steps being used as input
            channels (int): number of 3D variables. Default is 4 for our ERA5 configuration (U, V, T, and Q)
            surface_channels (int): number of surface (single-level) variables.
            input_only_channels (int): number of variables only used as input to the ML model (e.g., forcing variables)
            output_only_channels (int):number of variables that are only output by the model (e.g., diagnostic variables).
            levels (int): number of vertical levels for each 3D variable (should be the same across frames)
            dim (tuple): output dimensions of hidden state of each conv/transformer block in the encoder
            depth (tuple): number of attention blocks per encoder layer
            dim_head (int): dimension of each attention head.
            global_window_size (tuple): number of grid cells between cells in long range attention
            local_window_size (tuple): number of grid cells between cells in short range attention
            cross_embed_kernel_sizes (tuple): width of the cross embed kernels in each layer
            cross_embed_strides (tuple): stride of convolutions in each block
            attn_dropout (float): dropout rate for attention layout
            ff_dropout (float): dropout rate for feedforward layers.
            use_spectral_norm (bool): whether to use spectral normalization
            interp (bool): whether to use interpolation
            padding_conf (dict): padding configuration
            post_conf (dict): configuration for postblock processing
            **kwargs:
        """
        super().__init__()

        dim = tuple(dim)
        depth = tuple(depth)
        global_window_size = tuple(global_window_size)
        cross_embed_kernel_sizes = tuple([tuple(_) for _ in cross_embed_kernel_sizes])
        cross_embed_strides = tuple(cross_embed_strides)

        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        self.use_interp = interp
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = post_conf["activate"]

        # input channels
        self.input_only_channels = input_only_channels
        input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = input_channels

        # output channels
        output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = output_channels

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # dimensions
        last_dim = dim[-1]
        first_dim = input_channels if (patch_height == 1 and patch_width == 1) else dim[0]
        dims = [first_dim, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # allocate cross embed layers
        self.layers = nn.ModuleList([])

        # loop through hyperparameters
        for (
            dim_in,
            dim_out,
        ), num_layers, global_wsize, local_wsize, kernel_sizes, stride in zip(
            dim_in_and_out,
            depth,
            global_window_size,
            local_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            # create CrossEmbedLayer
            cross_embed_layer = CrossEmbedLayer(
                dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernel_sizes, stride=stride
            )

            # create Transformer
            transformer_layer = Transformer(
                dim=dim_out,
                local_window_size=local_wsize,
                global_window_size=global_wsize,
                depth=num_layers,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )

            # append everything
            self.layers.append(nn.ModuleList([cross_embed_layer, transformer_layer]))

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # define embedding layer using adjusted sizes
        # if the original sizes were good, adjusted sizes should == original sizes
        self.cube_embedding = CubeEmbedding(
            (frames, image_height, image_width),
            (frames, patch_height, patch_width),
            input_channels,
            dim[0],
        )

        # =================================================================================== #

        self.up_block1 = UpBlockPS(1 * last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlockPS(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlockPS(2 * (last_dim // 4), last_dim // 8, dim[0])
        # self.up_block4 = nn.ConvTranspose2d(2 * (last_dim // 8), output_channels, kernel_size=4, stride=2, padding=1)
        # self.up_block4 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        #     nn.Conv2d(2 * (last_dim // 8), output_channels, kernel_size=3, stride=1, padding=1),
        # )
        # replace your self.up_block4 = … block with this:
        scale = 2
        self.up_block4 = nn.Sequential(
            nn.Conv2d(
                2 * (last_dim // 8),  # in_channels
                self.output_channels * (scale**2),  # conv_out = target_channels * 4
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(upscale_factor=scale),  # now (target_channels, H*2, W*2),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
        )
        self.conv4up = self.up_block4[1]

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        if self.use_post_block:
            # freeze base model weights before postblock init
            if "skebs" in post_conf.keys():
                if post_conf["skebs"].get("activate", False) and post_conf["skebs"].get(
                    "freeze_base_model_weights", False
                ):
                    logger.warning("freezing all base model weights due to skebs config")
                    for param in self.parameters():
                        param.requires_grad = False

            logger.info("using postblock")
            self.postblock = PostBlock(post_conf)

    def forward(self, x):
        x_copy = None
        if self.use_post_block:  # copy tensor to feed into postBlock later
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:  # case where only using one time-step as input
            x = x.squeeze(2)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x

    def rk4(self, x):
        def integrate_step(x, k, factor):
            return self.forward(x + k * factor)

        k1 = self.forward(x)  # State at i
        k1 = torch.cat([x[:, :, -2:-1], k1], dim=2)
        k2 = integrate_step(x, k1, 0.5)  # State at i + 0.5
        k2 = torch.cat([x[:, :, -2:-1], k2], dim=2)
        k3 = integrate_step(x, k2, 0.5)  # State at i + 0.5
        k3 = torch.cat([x[:, :, -2:-1], k3], dim=2)
        k4 = integrate_step(x, k3, 1.0)  # State at i + 1

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6


if __name__ == "__main__":
    image_height = 640  # 640, 192
    image_width = 1280  # 1280, 288
    levels = 15
    frames = 2
    channels = 4
    surface_channels = 7
    input_only_channels = 3
    frame_patch_size = 2

    input_tensor = torch.randn(
        1,
        channels * levels + surface_channels + input_only_channels,
        frames,
        image_height,
        image_width,
    ).to("cuda")

    model = Camulator(
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        frame_patch_size=frame_patch_size,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        levels=levels,
        dim=(128, 256, 512, 1024),
        depth=(2, 2, 18, 2),
        global_window_size=(8, 4, 2, 1),
        local_window_size=5,
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(4, 2, 2, 2),
        attn_dropout=0.0,
        ff_dropout=0.0,
    ).to("cuda")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    y_pred = model(input_tensor.to("cuda"))
    print("Predicted shape:", y_pred.shape)
