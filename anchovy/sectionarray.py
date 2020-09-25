from math import inf

import numpy as np


class SectionArray:
    """Section coordinate array

    Parameters
    ----------
    station : array_like
        Station (lateral) coordinates. Must be in ascending order,
        one-dimensional, and the same size as `elevation`.
    elevation : array_like
        Elevation (vertical) coordinates. Must be one-dimensional
        and the same size as `station`.
    active_elev : float, optional
        Activation elevation for this section array. The default
        is ``-inf``. Results for computations at elevations below
        the activation elevation are returned as 0.

    """

    def __init__(self, station, elevation, active_elev=-inf):

        self._station = np.array(station)
        self._elevation = np.array(elevation)
        self._min_elevation = self._elevation.min()
        self._max_elevation = self._elevation.max()

        try:
            self._active_elev = float(active_elev)
        except TypeError as e:
            if e.args[0] == \
                    'only size-1 arrays can be converted to Python scalars':
                raise TypeError('activation elevation must be a scalar')
            else:
                raise

        if not np.all(np.isfinite(self._station)):
            raise ValueError("station must be finite")

        if not np.all(np.isfinite(self._elevation)):
            raise ValueError("elevation must be finite")

        if not self._station.ndim == 1:
            raise ValueError("station must be one dimensional")

        if not self._elevation.ndim == 1:
            raise ValueError("elevation must be one dimensional")

        if self._station.size != self._elevation.size:
            raise ValueError("station and elevation must have the same size")

        if not np.all(np.diff(self._station) >= 0):
            raise ValueError("station must be in ascending order")

    @staticmethod
    def _interp_elevation(s1, e1, s2, e2, s):
        slope = (e2 - e1)/(s2 - s1)
        return slope * (s - s2) + e2

    def _interp_station(self, s1, e1, s2, e2, e):

        slope = (s2 - s1)/(e2 - e1)
        return slope * (e - e1) + s1

    def _area(self, elevation):

        sub_array = self._sub_array(elevation, 'lr')
        nan_e = np.isnan(sub_array._elevation)
        return np.trapz(sub_array._station[~nan_e],
                        sub_array._elevation[~nan_e])

    def _array_comp(self, elevation, func, *args, **kwargs):

        elevation = np.array(elevation, dtype=np.float)
        val = np.empty_like(elevation)

        with np.nditer([elevation, val], [], [['readonly'], ['writeonly']]) \
                as it:
            for e, a in it:
                if e <= self._min_elevation or e < self._active_elev:
                    a[...] = 0
                elif not np.isfinite(e):
                    a[...] = np.nan
                else:
                    a[...] = func(e, *args, **kwargs)

        if elevation.ndim == 0:
            return float(val)
        else:
            return val

    def _perimeter(self, elevation, wall):

        sub_array = self._sub_array(elevation, wall)

        wp = 0

        for i in range(1, len(sub_array._station)):
            if np.isnan(sub_array._elevation[i-1]) \
                    or np.isnan(sub_array._elevation[i]):
                continue
            wp += \
                np.sqrt((sub_array._station[i]-sub_array._station[i-1])**2 +
                        (sub_array._elevation[i]-sub_array._elevation[i-1])**2)

        return wp

    def _sub_array(self, elevation, wall):
        """Computes and returns sub arrays from station and elevation

        Parameters
        ----------
        elevation : float
            Elevation for computing sub arrays
        wall : {None, 'l', 'r', 'lr'}
            Include wall above array ends

        Returns
        -------
        SectionArray

        """

        assert wall in [None, 'l', 'r', 'lr']

        if elevation <= self._min_elevation:
            return np.nan, np.nan

        s = []
        e = []

        if wall == 'l' or wall == 'lr':
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

            if len(e) > 0 and e2 > elevation and not np.isnan(e[-1]):
                s.append(s[-1])
                e.append(np.nan)

        if wall == 'r' or wall == 'lr':
            if elevation > self._elevation[-1]:
                s.append(self._station[-1])
                e.append(elevation)

        if np.isnan(e[-1]):
            s.pop()
            e.pop()

        cls = self.__class__
        result = cls.__new__(cls)
        result._station = np.array(s)
        result._elevation = np.array(e)

        return result

    def _top_width(self, elevation):
        sub_array = self._sub_array(elevation, 'lr')
        tw = 0
        for i in range(1, len(sub_array._station)):
            if np.isnan(sub_array._elevation[i-1]) \
                    or np.isnan(sub_array._elevation[i]):
                continue
            else:
                tw += sub_array._station[i] - sub_array._station[i-1]

        return tw

    def area(self, elevation):
        """Computes area of this array

        Parameters
        ----------
        elevation : array_like
            Elevation to compute area.

        Returns
        -------
        area : float or numpy.ndarray

        """

        return self._array_comp(elevation, self._area)

    def coordinates(self):
        """Returns copies of the coordinate arrays in this
        array

        Returns
        -------
        station, elevation : numpy.ndarray, numpy.ndarray

        """

        return self._station.copy(), self._elevation.copy()

    def copy(self):
        """Returns a copy of this array

        Returns
        -------
        array : SectionArray

        """

        return self.__class__(self._station, self._elevation)

    def max_elevation(self):
        """Returns the maximum elevation of this array

        Returns
        -------
        max_elevation : float

        """

        return self._max_elevation

    def max_station(self):
        """Returns the maximum station of this array

        Returns
        -------
        max_station : float

        """

        return self._station.max()

    def min_elevation(self):
        """Returns the minimum elevation of this array

        Returns
        -------
        min_elevation : float

        """

        return self._min_elevation

    def min_station(self):
        """Returns the minimum station of this array

        Returns
        -------
        min_station : float

        """

        return self._station.min()

    def perimeter(self, elevation, wall=None):
        """Returns the perimeter of this array

        Parameters
        ----------
        elevation : array_like
            Elevation for computing the perimeter.
        wall : {None, 'l', 'r', 'lr'}, optional
            If the elevation is greater than the elevation
            at the ends of this array, extend a wall to
            elevation in the computation of the perimeter.

        Returns
        -------
        perimeter : float or numpy.ndarray

        """

        if wall not in [None, 'l', 'r', 'lr']:
            raise ValueError("Invalid wall kwarg: {}".format(wall))

        args = [wall]
        return self._array_comp(elevation, self._perimeter, *args)

    def perimeter_array(self, elevation, wall=None):

        if wall not in [None, 'l', 'r', 'lr']:
            raise ValueError("Invalid wall kwarg: {}".format(wall))

        return self._sub_array(elevation, wall)

    def split(self, sect_stat, active_elev=None):
        """Creates subarrays of this section array based on
        sect_stat

        Parameters
        ----------
        sect_stat : array_like
            Subarray stationing
        active_elev : {None, array_like}, optional

        Returns
        -------
        list
            List of subarrays

        """

        split_station = []
        split_elevation = []

        j = 0
        s = []
        e = []

        sect_stat = np.array(sect_stat)

        if sect_stat.ndim == 0:
            sect_stat = sect_stat[np.newaxis]

        # loop through each station value in sect_stat
        for i in range(len(sect_stat)):

            # while the j-th station in this array is less than the i-th
            # section station, append it to the list and increment to the next
            # station in this array
            while self._station[j] < sect_stat[i]:
                s.append(self._station[j])
                e.append(self._elevation[j])
                j += 1

            s.append(sect_stat[i])  # append the i-th section station

            # if the last station added (sect_stat[i]) is equal to the next
            # station and the current elevation value is less than the next
            # elevation value
            if (j < len(self._station) - 1) and (s[-1] == self._station[j+1]) \
                    and (self._elevation[j] < self._elevation[j+1]):

                # add the
                e.append(self._elevation[j])
                s.append(s[-1])
                e.append(self._elevation[j+1])
                j += 1
            else:
                e_interp = self._interp_elevation(self._station[j-1],
                                                  self._elevation[j-1],
                                                  self._station[j],
                                                  self._elevation[j],
                                                  sect_stat[i])
                e.append(e_interp)

            split_station.append(np.array(s))
            split_elevation.append(np.array(e))

            if sect_stat[i] != self._station[j]:
                s = [split_station[-1][-1]]
                e = [split_elevation[-1][-1]]
            else:
                s = []
                e = []

        # finish the rest of the station, elevation arrays
        while j < len(self._station):
            s.append(self._station[j])
            e.append(self._elevation[j])
            j += 1

        split_station.append(np.array(s))
        split_elevation.append(np.array(e))

        if active_elev is None:
            active_elev = [self._active_elev] * len(split_station)

        return [SectionArray(s, e, a) for (s, e, a)
                in zip(split_station, split_elevation, active_elev)]

    def top_width(self, elevation):
        """Returns the top width of this array

        Parameters
        ----------
        elevation : array_like
            Elevation for computing the top width.

        Returns
        -------
        top_width : float or numpy.ndarray
        """

        return self._array_comp(elevation, self._top_width)
