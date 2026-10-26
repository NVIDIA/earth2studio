from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    data: np.ndarray,
    title: str,
    output_path: str | Path,
    variable_name: str = "t2m",
    units: str = "K",
    cmap: str = "viridis",
) -> Path:
    """Plots a 2D atmospheric field and saves to PNG with scientific labeling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # If 2D field
    if data.ndim > 2:
        field = data.squeeze()
        while field.ndim > 2:
            field = field[0]
    else:
        field = data

    im = ax.imshow(field, cmap=cmap, origin="upper", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.15, shrink=0.7)
    cbar.set_label(f"{variable_name.upper()} ({units})")

    ax.set_title(f"ORIGIN DATUM / EO-OBSCURA: {title}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
