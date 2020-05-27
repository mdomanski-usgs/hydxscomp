import numpy as np


class SubSection:
    """Cross section subsection

    Parameters
    ----------
    station : array_like
        Station (lateral) coordinate
    elevation : array_like
        Elevation (vertical) coordinate
    roughness : float
        Manning's roughness coefficient, in s/m**(1/3)

    """

    def __init__(self, station, elevation, roughness):

        self._station = np.array(station)
        self._elevation = np.array(elevation)
        self._roughness = float(roughness)
        self._min_elevation = self._elevation.min()

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

    def _area(self, elevation):

        s, e = self._wp(elevation)
        nan_e = np.isnan(e)
        return np.trapz(s[~nan_e], e[~nan_e])

    def _top_width(self, elevation):
        s, e = self._wp(elevation)
        tw = 0
        for i in range(1, len(s)):
            if np.isnan(e[i-1]) or np.isnan(e[i]):
                continue
            else:
                tw += s[i] - s[i-1]

        return tw

    def _array_comp(self, elevation, func):

        elevation = np.array(elevation, dtype=np.float)
        val = np.empty_like(elevation)

        with np.nditer([elevation, val], [], [['readonly'], ['writeonly']]) \
                as it:
            for e, a in it:
                if e <= self._min_elevation:
                    a[...] = 0
                elif not np.isfinite(e):
                    a[...] = np.nan
                else:
                    a[...] = func(e)

        if val.size == 1:
            return float(val)
        else:
            return val

    def _interp_station(self, s1, e1, s2, e2, e):

        slope = (s2 - s1)/(e2 - e1)
        return slope * (e - e1) + s1

    def _wp(self, elevation):
        """Station, elevation arrays of wetted perimeter"""

        if elevation <= self._min_elevation:
            return np.nan, np.nan

        s = []
        e = []

        if elevation > self._elevation[0]:
            s.append(self._station[0])
            e.append(elevation)

        if self._elevation[0] <= elevation:
            s.append(self._station[0])
            e.append(self._elevation[0])

        for i in range(1, len(self._station)):

            e1 = self._elevation[i - 1]
            e2 = self._elevation[i]

            s1 = self._station[i - 1]
            s2 = self._station[i]

            # in between the previous coordinate and this coordinate
            if (e1 < elevation and elevation < e2) or \
                    (e2 < elevation and elevation < e1):
                e.append(elevation)
                s.append(self._interp_station(s1, e1, s2, e2, elevation))

            # greater than or equal to this coordinate
            if e2 <= elevation:
                e.append(e2)
                s.append(s2)

            if e2 > elevation and not np.isnan(e[-1]):
                s.append(s[-1])
                e.append(np.nan)

        if elevation > self._elevation[-1]:
            s.append(self._station[-1])
            e.append(elevation)

        if np.isnan(e[-1]):
            s.pop()
            e.pop()

        return np.array(s), np.array(e)

    def area(self, elevation):
        """Computes wetted area of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations to comptue area

        Returns
        -------
        array_like
            Computed area

        """

        return self._array_comp(elevation, self._area)

    def top_width(self, elevation):
        """Computes top width of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations to compute top width

        Returns
        -------
        array_like
            Computed top width

        """

        return self._array_comp(elevation, self._top_width)
