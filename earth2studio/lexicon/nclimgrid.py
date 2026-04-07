from collections.abc import Callable

import numpy as np

from earth2studio.lexicon.base import LexiconType


class NClimGridLexicon(metaclass=LexiconType):
    """
    NClimGrid gridded dataset lexicon.

    Provides canonical → native variable mapping and unit normalization
    for the CONUS NClimGrid Zarr dataset.

    Dataset characteristics
    -----------------------
    • Spatial coverage: CONUS (~0.0417° grid)
    • Temporal coverage: daily (1952–present)
    • Variables are already physically meaningful fields (not station codes)
    • Missing values may exist (NaN / masked)

    Unit conventions
    ----------------
    • Temperature fields stored in °C → converted to Kelvin
    • Precipitation stored in mm → converted to meters
    • SPI is dimensionless → unchanged

    Design goals
    ------------
    • Future extensibility (new variables / datasets)
    • Robust unit normalization
    • Strong validation behaviour
    • Explicit metadata layer
    """

    # -------------------------------------------------------
    # Rich variable metadata structure (future-proof)
    # -------------------------------------------------------

    META: dict[str, dict] = {
        "t2m_max": {
            "native": "tmax",
            "units_native": "degC",
            "units_e2s": "K",
            "description": "daily maximum temperature at 2m",
        },
        "t2m_min": {
            "native": "tmin",
            "units_native": "degC",
            "units_e2s": "K",
            "description": "daily minimum temperature at 2m",
        },
        "tp": {
            "native": "prcp",
            "units_native": "mm",
            "units_e2s": "m",
            "description": "daily total precipitation",
        },
        "spi": {
            "native": "spi",
            "units_native": "dimensionless",
            "units_e2s": "dimensionless",
            "description": "standardized precipitation index",
        },
    }

    # Canonical vocabulary (required by LexiconType)
    VOCAB: dict[str, str] = {
        k: f"{v['description']} ({v['units_e2s']})" for k, v in META.items()
    }

    # -------------------------------------------------------
    # Strong validation + modifier factory
    # -------------------------------------------------------

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """
        Resolve canonical variable.

        Returns
        -------
        native_variable_name, modifier_function
        """

        if val not in cls.META:
            raise KeyError(
                f"NClimGridLexicon: unknown variable '{val}'. "
                f"Valid variables: {list(cls.META.keys())}"
            )

        meta = cls.META[val]
        native = meta["native"]

        # ---------------------------------------------------
        # Robust modifier (handles NaN + dtype normalization)
        # ---------------------------------------------------

        def modifier(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype="float32")

            if native in ("tmax", "tmin"):
                return x + 273.15

            if native == "prcp":
                return x / 1000.0

            # SPI / future dimensionless variables
            return x

        return native, modifier
