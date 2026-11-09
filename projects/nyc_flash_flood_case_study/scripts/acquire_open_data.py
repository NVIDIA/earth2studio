from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"


DATASET_SPECS = {
    "nyc_boundary": {
        "url": "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON",
        "path": DATA_DIR / "nyc_boundary.geojson",
    },
    "study_area": {
        "url": "https://example.invalid/study_area.geojson",
        "path": DATA_DIR / "study_area.geojson",
    },
    "impervious_surface": {
        "url": "https://example.invalid/nlcd_impervious_surface.tif",
        "path": DATA_DIR / "landcover" / "impervious_surface.tif",
    },
    "dem": {
        "url": "https://example.invalid/usgs_dem.tif",
        "path": DATA_DIR / "terrain" / "dem.tif",
    },
    "flood_reports": {
        "url": "https://example.invalid/flood_reports.geojson",
        "path": DATA_DIR / "impacts" / "flood_reports.geojson",
    },
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with dest.open("wb") as handle:
        handle.write(response.content)


def acquire_all() -> None:
    for name, spec in DATASET_SPECS.items():
        if spec["path"].exists():
            continue
        print(f"Downloading {name} from {spec['url']}")
        download_file(spec["url"], spec["path"])


if __name__ == "__main__":
    acquire_all()
