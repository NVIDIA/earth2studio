import torch
import numpy as np

from collections import OrderedDict
from datetime import datetime
from earth2studio.models.auto import Package
from earth2studio.models.px import CBottleVideo
from contextlib import nullcontext
import nvtx
from tqdm import tqdm


start_time = datetime(1990,1,1)
HPX_LEVEL=6

def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()

with cuda_profiler():
    with profiler_emit_nvtx():
        # model = CBottleVideo.load_model(CBottleVideo.load_default_package(), seed=1).to("cuda")
        with nvtx.annotate("load model", color="blue"):
            model = CBottleVideo.load_model(CBottleVideo.load_default_package(), lat_lon=False, seed=1).to("cuda")
        coords = OrderedDict({
            "time": np.array([start_time], dtype=np.datetime64),
            "lead_time": np.array([np.timedelta64(0, 'h')]),
            "variable": np.array(model.VARIABLES),
            "hpx": np.arange(4**HPX_LEVEL * 12),
            # "lat": np.linspace(90, -90, 721),
            # "lon": np.linspace(0, 360, 1440, endpoint=False),
        })
        # inputs = torch.full((1,1,45,721,1440), float('nan')).cuda()
   
        # Create timer objects only if CUDA is available
        use_cuda_timing = torch.cuda.is_available()
        if use_cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        else:
            # Dummy no-op functions for CPU case
            class DummyEvent:
                def record(self):
                    pass

                def synchronize(self):
                    pass

                def elapsed_time(self, _):
                    return 0

            start = end = DummyEvent() 
            
        iterations = 10
        warmups = 2
        for i in tqdm(range(iterations), desc="Running inference"):
            if i == warmups:
                start.record()
            if i == 5:
                print(f"Starting Profiler at {i}")
                cuda_profiler_start()

            if i == 10:
                print(f"Stopping Profiler at {i}")
                cuda_profiler_stop()
                
            inputs = torch.full((1,1,45,49152), float('nan')).cuda()

            # Execute prognostic
            outputs = []
            with nvtx.annotate("model forward", color="blue"):
                model_itr = model.create_iterator(inputs, coords)
            for step, (x, coords) in enumerate(model_itr): 
                print(x.shape)
                outputs.append(x)
                if step == 11:
                    break
            out = torch.cat(outputs, dim=1).cpu().numpy()
            
        end.record()
        end.synchronize()
        elapsed_time = (
            start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0
        )  # Convert ms to s
        if use_cuda_timing:
            print(
                f"average time to generate one sample = {elapsed_time/(iterations-warmups)} s"
            )

# breakpoint()

# # Post processing!
# #HPX
# def prepare(x):
#     ring_order = healpix.reorder(
#         x,
#         earth2grid.healpix.PixelOrder.NEST,
#         earth2grid.healpix.PixelOrder.RING,
#     )
#     return {
#         test_dataset.batch_info.channels[c]: ring_order[:, c].cpu()
#         for c in range(x.shape[1])
#     }
            
# def diagnostics(pred, lr, target, output_path):
#     titles = ["input", "prediction", "target"]
#     for var in pred.keys():
#         plt.figure(figsize=(50, 25))
#         vmin = torch.min(pred[var][0, 0])
#         vmax = torch.max(pred[var][0, 0])
#         for idx, data, title in zip(
#             np.arange(1, 4), [lr[var][0, 0], pred[var][0, 0], target[var][0, 0]], titles
#         ):
#             visualizations.visualize(
#                 data,
#                 pos=(1, 3, idx),
#                 title=title,
#                 nlat=721,
#                 nlon=1440,
#                 vmin=vmin,
#                 vmax=vmax,
#             )
#         plt.tight_layout()
#         plt.savefig(f"{output_path}/output_{var}")

# import matplotlib.pyplot as plt
# vidx = -13
# n_frames = out.shape[1]

# fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# axes = axes.flatten()

# for i in range(n_frames):
#     axes[i].imshow(out[0,i,vidx,:])
#     axes[i].set_title(f'Frame {i}')
#     axes[i].axis('off')

# plt.tight_layout()
# plt.savefig("cbottle.jpg")