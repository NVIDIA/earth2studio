"""Download the small CAMulator assets used by compare_physics.py / run_real.py
into ./assets (pinned HF revision). The 4.8 GB checkpoint is NOT fetched here;
CAMulator.load_default_package() handles it."""
import os

from huggingface_hub import hf_hub_download

from earth2studio.data.camulator import CAMULATOR_HF_REVISION, HF_REPO_ID
from earth2studio.models.px.camulator import _MEAN_FILE, _PHYSICS_FILE, _STD_FILE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAMULATOR_ASSETS", os.path.join(HERE, "assets"))
os.makedirs(OUT, exist_ok=True)
FILES = {
    "mean.nc": _MEAN_FILE,
    "std.nc": _STD_FILE,
    "statics_b.nc": _PHYSICS_FILE,
    "ic_1981.pth": "initial_conditions/init_camulator_condition_tensor_1981-01-01T00Z.pth",
}
for short, remote in FILES.items():
    src = hf_hub_download(HF_REPO_ID, remote, revision=CAMULATOR_HF_REVISION)
    dst = os.path.join(OUT, short)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(short, "->", src)
