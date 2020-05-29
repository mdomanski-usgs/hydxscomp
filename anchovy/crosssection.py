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

        s, e = self._wa_array(elevation)
        nan_e = np.isnan(e)
        return np.trapz(s[~nan_e], e[~nan_e])

    def _top_width(self, elevation):
        s, e = self._wa_array(elevation)
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

    def _wetted_perimeter(self, elevation):

        s, e = self._wp_array(elevation)

        wp = 0

        for i in range(1, len(s)):
            if np.isnan(e[i-1]) or np.isnan(e[i]):
                continue
            wp += np.sqrt((s[i]-s[i-1])**2 + (e[i]-e[i-1])**2)

        return wp

    def _wa_array(self, elevation):
        """Station, elevation arrays of wetted area"""

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

    def _wp_array(self, elevation):
        """Station, elevation arrays of wetted perimeter"""

        if elevation <= self._min_elevation:
            return np.nan, np.nan

        s = []
        e = []

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

        if np.isnan(e[-1]):
            s.pop()
            e.pop()

        return np.array(s), np.array(e)

    def area(self, elevation):
        """Computes wetted area of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing area

        Returns
        -------
        area : float or ndarray

        """

        return self._array_comp(elevation, self._area)

    def top_width(self, elevation):
        """Computes top width of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing top width

        Returns
        -------
        top_width : float or ndarray

        """

        return self._array_comp(elevation, self._top_width)

    def wetted_perimeter(self, elevation):
        """Computes wetted perimeter of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing wetted perimeter

        Returns
        -------
        wetted_perimeter : float or ndarray

        """

        return self._array_comp(elevation, self._wetted_perimeter)


class CrossSection:
    """Hydraulic cross section

    Parameters
    ----------
    station : array_like
        Station (lateral) coordinates. Must be in ascending order,
        one-dimensional, and the same size as `elevation`.
    elevation : array_like
        Elevation (vertical) coordinates. Must be one-dimensional
        and the same size as `station`.
    roughness : array_like
        Roughness values of subsections.
    sect_stat : array_like, optional
        Station values that define section splits. The default is
        None. Must not be None if the size of `roughness` is more
        than one. Values must be within the range of `station`
        (exclusive). The number of elements must be one less than
        the number of elements in `roughness`.

    """

    def __init__(self, station, elevation, roughness, sect_stat=None):

        self._station = np.array(station, dtype=np.float)
        self._elevation = np.array(elevation, dtype=np.float)

        roughness = np.array(roughness, dtype=np.float)

        if sect_stat is not None:
            sect_stat = np.array(sect_stat, dtype=np.float)

        if not np.all(np.isfinite(self._station)):
            raise ValueError("station must be finite")

        if not np.all(np.isfinite(self._elevation)):
            raise ValueError("elevation must be finite")

        if not np.all(np.isfinite(roughness)):
            raise ValueError("roughness must be finite")

        if not self._station.ndim == 1:
            raise ValueError("station must be one dimensional")

        if not self._elevation.ndim == 1:
            raise ValueError("elevation must be one dimensional")

        if self._station.size != self._elevation.size:
            raise ValueError("station and elevation must have the same size")

        if not np.all(np.diff(self._station) >= 0):
            raise ValueError("station must be in ascending order")

        if roughness.size > 1:
            if sect_stat is None:
                raise ValueError("rough_stat cannot be None")
            if roughness.size - 1 != sect_stat.size:
                raise ValueError("Invalid number of rough_stat values")
            if sect_stat.min() <= self._station.min() \
                    or self._station.max() <= sect_stat.max():
                raise ValueError(
                    "rough_stat bounds must be inside station bounds")
            if sect_stat.size > 1:
                if not sect_stat.ndim == 1:
                    raise ValueError("rough_stat must be one dimensional")
                if not np.all(np.diff(sect_stat) > 0):
                    raise ValueError("rough_stat must be in ascending order")

        if roughness.size > 1:
            self._subsections = self._sections(
                station, elevation, roughness, sect_stat)
        else:
            self._subsections = [SubSection(station, elevation, roughness)]

    @staticmethod
    def _interp_elevation(s1, e1, s2, e2, s):
        slope = (e2 - e1)/(s2 - s1)
        return slope * (s - s2) + e2

    def _sections(self, station, elevation, roughness, rough_stat):

        sections = []

        split_station, split_elevation = self._split_arrays(
            station, elevation, rough_stat)

        for i, n in enumerate(roughness):
            sections.append(SubSection(
                split_station[i], split_elevation[i], n))

        return sections

    @classmethod
    def _split_arrays(cls, station, elevation, sect_stat):

        split_station = []
        split_elevation = []

        j = 0
        s = []
        e = []

        if sect_stat.ndim == 0:
            sect_stat = sect_stat[np.newaxis]

        for i in range(len(sect_stat)):

            while station[j] < sect_stat[i]:
                s.append(station[j])
                e.append(elevation[j])
                j += 1

            s.append(sect_stat[i])
            e_interp = cls._interp_elevation(station[j-1], elevation[j-1],
                                             station[j], elevation[j],
                                             sect_stat[i])
            e.append(e_interp)

            split_station.append(np.array(s))
            split_elevation.append(np.array(e))

            if sect_stat[i] != station[j]:
                s = [split_station[-1][-1]]
                e = [split_elevation[-1][-1]]
            else:
                s = []
                e = []

        # finish the rest of the station, elevation arrays
        while j < len(station):
            s.append(station[j])
            e.append(elevation[j])
            j += 1

        split_station.append(np.array(s))
        split_elevation.append(np.array(e))

        return split_station, split_elevation

    def area(self, elevation):
        """Area computed for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing area.

        Returns
        -------
        area : float or ndarray

        """

        area = 0

        for ss in self._subsections:
            area += ss.area(elevation)

        return area

    def top_width(self, elevation):
        """Computes top width for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing top width.

        Returns
        -------
        top_width : float or ndarray

        """

        top_width = 0

        for ss in self._subsections:
            top_width += ss.top_width(elevation)

        return top_width

    def wetted_perimeter(self, elevation):
        """Computes the wetted perimeter for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing wetted perimeter.

        Returns
        -------
        wetted_perimeter : float or ndarray

        """

        wp = 0

        for ss in self._subsections:
            wp += ss.wetted_perimeter(elevation)

        return wp
