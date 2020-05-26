import numpy as np


class SubSection:
    """Cross section subsection

    Parameters
    ----------
    station : array_like
    elevation : array_like
    roughness : float
        Manning's roughness coefficient, in s/m**(1/3)

    """

    def __init__(self, station, elevation, roughness):

        self._station = np.array(station)
        self._elevation = np.array(elevation)
        self._roughness = float(roughness)

        if not np.all(np.isfinite(self._station)):
            raise ValueError("station must be finite")

        if not np.all(np.isfinite(self._elevation)):
            raise ValueError("elevation must be finite")

        if not np.isfinite(self._roughness):
            raise ValueError("roughness must be finite")

        if not self._station.ndim == 1:
            raise ValueError("station must be one dimensional")

        if not self._elevation.ndim == 1:
            raise ValueError("elevation must be one dimensional")

        if self._station.size != self._elevation.size:
            raise ValueError("station and elevation must have the same size")

        if not np.all(np.diff(self._station) >= 0):
            raise ValueError("station must be in ascending order")
