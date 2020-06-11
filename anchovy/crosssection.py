import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

    """

    def __init__(self, station, elevation):

        self._station = np.array(station)
        self._elevation = np.array(elevation)
        self._min_elevation = self._elevation.min()
        self._max_elevation = self._elevation.max()

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

        sub_array = self._sub_array(elevation, 'a')
        nan_e = np.isnan(sub_array._elevation)
        return np.trapz(sub_array._station[~nan_e],
                        sub_array._elevation[~nan_e])

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

        if elevation.ndim == 0:
            return float(val)
        else:
            return val

    def _perimeter(self, elevation):
        sub_array = self._sub_array(elevation, 'p')

        wp = 0

        for i in range(1, len(sub_array._station)):
            if np.isnan(sub_array._elevation[i-1]) \
                    or np.isnan(sub_array._elevation[i]):
                continue
            wp += \
                np.sqrt((sub_array._station[i]-sub_array._station[i-1])**2 +
                        (sub_array._elevation[i]-sub_array._elevation[i-1])**2)

        return wp

    def _sub_array(self, elevation, array_type):
        """Computes and returns sub arrays from station and elevation

        Parameters
        ----------
        elevation : float
            Elevation for computing sub arrays
        array_type : {'p', 'a'}
            Type of computation. Wetted perimeter or wetted area.

        Returns
        -------
        SectionArray

        """

        if array_type != 'p' and array_type != 'a':
            raise ValueError("Invalid array type: {}".format(array_type))

        if elevation <= self._min_elevation:
            return np.nan, np.nan

        s = []
        e = []

        if array_type == 'a':
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

        if array_type == 'a':
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
        sub_array = self._sub_array(elevation, 'a')
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

    def perimeter(self, elevation):
        """Returns the perimeter of this array

        Parameters
        ----------
        elevation : array_like
            Elevation for computing the perimeter.

        Returns
        -------
        perimeter : float or numpy.ndarray

        """

        return self._array_comp(elevation, self._perimeter)

    def perimeter_array(self, elevation):

        return self._sub_array(elevation, 'p')

    def split(self, sect_stat):

        split_station = []
        split_elevation = []

        j = 0
        s = []
        e = []

        if sect_stat.ndim == 0:
            sect_stat = sect_stat[np.newaxis]

        for i in range(len(sect_stat)):

            while self._station[j] < sect_stat[i]:
                s.append(self._station[j])
                e.append(self._elevation[j])
                j += 1

            s.append(sect_stat[i])
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

        return [SectionArray(s, e) for (s, e)
                in zip(split_station, split_elevation)]

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


class SubSection:
    """Cross section subsection

    Parameters
    ----------
    section_array : SectionArray
    roughness : float
        Manning's roughness coefficient, in s/m**(1/3)

    """

    def __init__(self, section_array, roughness):

        self._array = section_array.copy()

        self._roughness = float(roughness)
        self._min_elevation = self._array.min_elevation()

        if not np.isfinite(self._roughness):
            raise ValueError("roughness must be finite")

    def area(self, elevation):
        """Computes wetted area of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing area

        Returns
        -------
        area : float or numpy.ndarray

        """

        return self._array.area(elevation)

    def conveyance(self, elevation):
        """Computes conveyance for this subsection

        Parameters
        ----------
        elevation : array_like
            Elevation for computing conveyance.

        Returns
        -------
        conveyance : float or numpy.ndarray

        """

        area = self.area(elevation)
        hydraulic_radius = self.hydraulic_radius(elevation)

        return 1.486/self._roughness * hydraulic_radius**(2/3) * area

    def hydraulic_radius(self, elevation):
        """Computes hydraulic radius for this subsection

        Parameters
        ----------
        elevation : array_like
            Elevation for computing hydraulic radius.

        Returns
        -------
        hydraulic_radius : float or numpy.ndarray

        """

        area = self.area(elevation)
        wetted_perimeter = self.wetted_perimeter(elevation)

        hydraulic_radius = np.zeros_like(elevation)
        zeros = (area == 0) & (wetted_perimeter == 0)
        hydraulic_radius[~zeros] = area[~zeros]/wetted_perimeter[~zeros]

        return hydraulic_radius

    def top_width(self, elevation):
        """Computes top width of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing top width

        Returns
        -------
        top_width : float or numpy.ndarray

        """

        return self._array.top_width(elevation, self._array.top_width)

    def wetted_perimeter(self, elevation):
        """Computes wetted perimeter of this subsection

        Parameters
        ----------
        elevation : array_like
            Elevations for computing wetted perimeter

        Returns
        -------
        wetted_perimeter : float or numpy.ndarray

        """

        return self._array.perimeter(elevation)


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

        self._array = SectionArray(station, elevation)

        roughness = np.array(roughness, dtype=np.float)

        if sect_stat is not None:
            sect_stat = np.array(sect_stat, dtype=np.float)

        if not np.all(np.isfinite(roughness)):
            raise ValueError("roughness must be finite")

        if roughness.size > 1:
            if sect_stat is None:
                raise ValueError("rough_stat cannot be None")
            if roughness.size - 1 != sect_stat.size:
                raise ValueError("Invalid number of rough_stat values")
            if sect_stat.min() <= self._array.min_station() \
                    or self._array.max_station() <= sect_stat.max():
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
            array = SectionArray(station, elevation)
            self._subsections = [SubSection(array, roughness)]

        self._sect_stat = sect_stat

    def _sections(self, station, elevation, roughness, rough_stat):

        sections = []

        split_arrays = self._array.split(rough_stat)

        for i, n in enumerate(roughness):
            sections.append(SubSection(split_arrays[i], n))

        return sections

    def area(self, elevation):
        """Computes area for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing area.

        Returns
        -------
        area : float or numpy.ndarray

        """

        return self._array.area(elevation)

    def conveyance(self, elevation):
        """Computes conveyance for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation to compute conveyance.

        Returns
        -------
        conveyance : float or numpy.ndarray

        """

        conveyance = 0

        for ss in self._subsections:
            conveyance += ss.conveyance(elevation)

        return conveyance

    def coordinates(self):
        """Returns a copy of the coordinates in this cross section

        Returns
        -------
        numpy.ndarray, numpy.ndarray : station, elevation

        """

        return self._array.coordinates()

    def hydraulic_depth(self, elevation):
        """Computes hydraulic depth for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing hydraulic depth.

        Returns
        -------
        hydraulic_depth : float or numpy.ndarray

        """

        area = self.area(elevation)
        top_width = self.top_width(elevation)

        hydraulic_depth = np.zeros_like(elevation)
        zeros = (area == 0) & (top_width == 0)
        hydraulic_depth[~zeros] = area[~zeros]/top_width[~zeros]

        return hydraulic_depth

    def hydraulic_radius(self, elevation):
        """Computes hydraulic radius for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing hydraulic radius.

        Returns
        -------
        hydraulic_radius : float or numpy.ndarray

        """

        area = self.area(elevation)
        wetted_perimeter = self.wetted_perimeter(elevation)

        hydraulic_radius = np.zeros_like(elevation)
        zeros = (area == 0) & (wetted_perimeter == 0)
        hydraulic_radius[~zeros] = area[~zeros]/wetted_perimeter[~zeros]

        return hydraulic_radius

    def plot(self, elevation=None, ax=None):
        """Plots this cross section

        Parameters
        ----------
        elevation : float or None, optional
            Elevation for plotting characteristics. The default is
            None, which doesn't plot characteristics.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. The default is None, which creates a
            new axes. If not None, `ax` is returned.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """

        s, e = self._array.coordinates()

        if ax is None:

            ax = plt.axes()

        ax.plot(s, e, 'k', marker='.', label='Coordinates')

        if elevation is not None:

            elevation = float(elevation)

            if elevation > self._array.min_elevation():

                wp = self._array.perimeter_array(elevation)
                wp_s, wp_e = wp.coordinates()
                plt.plot(wp_s, wp_e, 'g', linewidth=5,
                         label='Wetted perimeter')

                e_nan = np.isnan(wp_e)
                tw_e = elevation*np.ones_like(wp_e)
                tw_e[e_nan] = np.nan
                plt.plot(wp_s, tw_e, 'b', linewidth=2.5, label='Top width')

                xs_area_zy = [*zip(wp_s, wp_e)]

                if elevation > wp_e[0]:
                    xs_area_zy.insert(0, (wp_s[0], elevation))
                if elevation > wp_e[-1]:
                    xs_area_zy.append((wp_s[-1], elevation))

                if len(xs_area_zy) > 2:
                    poly = Polygon(xs_area_zy, facecolor='b',
                                   alpha=0.25, label='Wetted area')
                    ax.add_patch(poly)

        if self._sect_stat is not None:
            sect_elev = np.interp(self._sect_stat, s, e)
            ax.plot(self._sect_stat, sect_elev, linestyle='None',
                    marker='s', markerfacecolor='r', markeredgecolor='r',
                    label='Sub section')

        ax.legend()
        ax.set_xlabel('Station, in ft')
        ax.set_ylabel('Elevation, in ft')

        return ax

    def top_width(self, elevation):
        """Computes top width for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing top width.

        Returns
        -------
        top_width : float or numpy.ndarray

        """

        return self._array.top_width(elevation)

    def velocity_coeff(self, elevation):
        """Computes velocity coefficient (alpha) for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevations to compute velocity coefficient.

        Returns
        -------
        velocity_coeff : float or numpy.ndarray

        """

        elevation = np.array(elevation, dtype=np.float)
        elevation_dims = elevation.ndim
        if elevation_dims == 0:
            elevation = elevation[np.newaxis]

        sum = np.zeros_like(elevation)

        for ss in self._subsections:
            a_ss = ss.area(elevation)
            k_ss = ss.conveyance(elevation)

            zero_a = a_ss == 0

            if np.any(~zero_a):
                sum[~zero_a] += (k_ss[~zero_a]**3)/(a_ss[~zero_a]**2)

        area = self.area(elevation)
        k_t = self.conveyance(elevation)

        zero_k_t = k_t == 0
        velocity_coeff = np.zeros_like(elevation)
        velocity_coeff[zero_k_t] = np.nan
        velocity_coeff[~zero_k_t] = area[~zero_k_t]**2 / \
            k_t[~zero_k_t]**3*sum[~zero_k_t]

        if elevation_dims == 0:
            return float(velocity_coeff)
        else:
            return velocity_coeff

    def wetted_perimeter(self, elevation):
        """Computes the wetted perimeter for this cross section

        Parameters
        ----------
        elevation : array_like
            Elevation for computing wetted perimeter.

        Returns
        -------
        wetted_perimeter : float or numpy.ndarray

        """

        return self._array.perimeter(elevation)
