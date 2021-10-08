from math import inf
import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
import numpy as np

import hydxscomp
from hydxscomp.sectionarray import SectionArray

logger = hydxscomp.logger.getChild(__name__)


class SubSection:
    """Cross section subsection

    Parameters
    ----------
    section_array : SectionArray
    frict : Frict
        Manning's n relation
    wall : {None, 'l', 'r', 'lr'}, optional
    xs : CrossSection, optional
        Parent cross section object

    """

    def __init__(self, section_array, frict, wall=None, xs=None):

        if xs is None:
            self.logger = logger.getChild(self.__class__.__name__)
        else:
            self.logger = xs.logger.getChild(self.__class__.__name__)

        self._frict = frict

        if wall not in [None, 'l', 'r', 'lr']:
            raise ValueError("Invalid value for wall: {}".format(wall))
        self._wall = wall

        self._array = section_array.copy()

        self._min_elevation = self._array.min_elevation()

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

    def array(self):
        """Returns a copy of the section array of this
        subsection

        Returns
        -------
        SectionArray

        """

        return self._array.copy()

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
        roughness = self._frict.roughness(elevation)

        return 1.486/roughness * hydraulic_radius**(2/3) * area

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

        if isinstance(area, float) and isinstance(wetted_perimeter, float):
            if area != 0 and wetted_perimeter != 0:
                hydraulic_radius = area/wetted_perimeter
            else:
                hydraulic_radius = 0
        else:
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

        return self._array.perimeter(elevation, self._wall)


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
    frict : Frict
        Elevation/roughness relation
    sect_stat : array_like, optional
        Station values that define section splits. The default is
        None. Must not be None if the size of `roughness` is more
        than one. Values must be within the range of `station`
        (exclusive).
    active_elev : array_like, optional
        Activation elevation for each subsection. If None, -inf is
        used.
    wall : boolean, optional
        Include a vertical wall with friction properties in the
        computation of the wetted perimeter when the elevation
        exceeds the cross section geometry elevation on the
        sides. If True, a vertical wall with friction is
        included in the computation.

    """

    def __init__(self, station, elevation, frict, sect_stat=None,
                 active_elev=None, wall=False):

        self.logger = logger.getChild(self.__class__.__name__)

        self._array = SectionArray(station, elevation)
        self._wall = bool(wall)

        if sect_stat is not None:
            sect_stat = np.array(sect_stat, dtype=np.float64)

        self._frict = frict

        if sect_stat is not None:

            if sect_stat.min() <= self._array.min_station() \
                    or self._array.max_station() <= sect_stat.max():
                raise ValueError(
                    "sect_stat bounds must be inside station bounds")

            if sect_stat.size > 1:
                if not sect_stat.ndim == 1:
                    raise ValueError("sect_stat must be one dimensional")
                if not np.all(np.diff(sect_stat) > 0):
                    raise ValueError("sect_stat must be in ascending order")

            if active_elev is None:
                active_elev = np.full(
                    (sect_stat.size + 1,), -np.inf, dtype=np.float64)
            else:
                active_elev = np.array(active_elev, dtype=np.float64)

            self._subsections = \
                self._sections(self._array, self._frict,
                               sect_stat, active_elev, wall)
        else:
            if active_elev is None:
                active_elev = -inf
            array = SectionArray(station, elevation, active_elev)
            if self._wall:
                self._subsections = [SubSection(array, self._frict, 'lr')]
            else:
                self._subsections = [SubSection(array, self._frict)]

        self._sect_stat = sect_stat

    @staticmethod
    def _plot_subsection(subsection, elevation, wall, ax):

        array = subsection.array()
        wp = array.perimeter_array(elevation, wall)
        wp_s, wp_e = wp.coordinates()

        try:
            if (np.isnan(wp_s) or np.isnan(wp_e)):
                return
        except ValueError:
            pass

        ax.plot(wp_s, wp_e, 'g', linewidth=5)

        e_nan = np.isnan(wp_e)
        tw_e = elevation*np.ones_like(wp_e)
        tw_e[e_nan] = np.nan
        ax.plot(wp_s, tw_e, 'b', linewidth=2.5)

        xs_area_zy = [*zip(wp_s, wp_e)]

        if elevation > wp_e[0]:
            xs_area_zy.insert(0, (wp_s[0], elevation))
        if elevation > wp_e[-1]:
            xs_area_zy.append((wp_s[-1], elevation))

        if len(xs_area_zy) > 2:
            poly = Polygon(xs_area_zy, facecolor='b', alpha=0.25)
            ax.add_patch(poly)

    def _sections(self, array, frict, rough_stat,
                  active_elev, wall):

        sections = []

        split_arrays = array.split(rough_stat, active_elev)

        n_sections = len(split_arrays)

        # for i, n in enumerate(roughness):
        for i, array in enumerate(split_arrays):
            if i == 0 and wall:
                sections.append(SubSection(
                    split_arrays[i], frict, wall='l', xs=self))
            elif i == n_sections - 1 and wall:
                sections.append(SubSection(
                    split_arrays[i], frict, wall='r', xs=self))
            else:
                sections.append(SubSection(split_arrays[i], frict, xs=self))

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

        elevation = np.array(elevation, dtype=float)
        area = np.zeros_like(elevation)
        for ss in self._subsections:
            area += ss.area(elevation)

        if elevation.shape == ():
            return float(area)
        else:
            return area

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

    def n_subsections(self):
        """Returns the number of subsections in this cross section

        Returns
        -------
        n_subsections : int
            Number of subsections

        """

        return len(self._subsections)

    def plot(self, elevation=None, ax=None, legend=True):
        """Plots this cross section

        Parameters
        ----------
        elevation : float or None, optional
            Elevation for plotting characteristics. The default is
            None, which doesn't plot characteristics.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. The default is None, which creates a
            new axes. If not None, `ax` is returned.
        legend : bool, optional
            Show legend in plot axes. The default is True.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """

        s, e = self._array.coordinates()

        if ax is None:
            ax = plt.axes()

        # list of handles for legend
        handles = []

        # show left or right walls
        if self._wall:
            l_wall = 'l'
            r_wall = 'r'
            lr_wall = 'lr'
        else:
            l_wall = None
            r_wall = None
            lr_wall = None

        # if an elevation is provided, plot the wetted area, top width, and
        # wetted perimeter of each subsection
        if elevation is not None:

            elevation = float(elevation)

            n_ss = len(self._subsections)

            if n_ss == 1:
                self._plot_subsection(
                    self._subsections[0], elevation, lr_wall, ax)
            else:
                for i, ss in enumerate(self._subsections):
                    if i == 0:
                        self._plot_subsection(ss, elevation, l_wall, ax)
                    elif i == n_ss - 1:
                        self._plot_subsection(ss, elevation, r_wall, ax)
                    else:
                        self._plot_subsection(ss, elevation, None, ax)

            # create proxy artists to add to the legend
            area_patch = Patch(color='blue', alpha=0.25, label='Wetted area')
            handles.append(area_patch)

            tw_line = Line2D([], [], color='blue',
                             linewidth=2.5, label='Top width')
            handles.append(tw_line)

            wp_line = Line2D([], [], color='green',
                             linewidth=5, label='Wetted perimeter')
            handles.append(wp_line)

        # plot the coordinates
        for ss in self._subsections:
            s, e = ss.array().coordinates()
            ax.plot(s, e, 'k', marker='.')

        coord_line = Line2D([], [], color='k', marker='.', label='Coordinates')
        handles.append(coord_line)

        # plot the points where subsections are divided
        if len(self._subsections) > 1:
            s, e = self._subsections[0].array().coordinates()
            sect_elev = [e[-1]]
            sect_station = [s[-1]]
            for i in range(1, len(self._subsections)):
                s, e = self._subsections[i].array().coordinates()
                sect_station.append(s[0])
                sect_elev.append(e[0])
            ss_point = ax.plot(sect_station, sect_elev, linestyle='None',
                               marker='s', markerfacecolor='r',
                               markeredgecolor='r', label='Sub-section station')
            handles.append(ss_point[0])

        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
            ax.legend(handles=handles, bbox_to_anchor=(
                1, 0.5), loc='center left')

        ax.set_xlabel('Station, in ft')
        ax.set_ylabel('Elevation, in ft')

        return ax

    def roughness(self, elevation):
        """Computes roughness weighted by wetted perimeter for this
        cross section

        Parameters
        ----------
        elevation : array_like
            Elevations for computing roughness

        Returns
        -------
        roughness : float or numpy.ndarray

        """

        return self._frict.roughness(elevation)

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

    def vel_coeff(self, elevation):
        """Computes velocity coefficient (alpha) for this cross
        section

        Parameters
        ----------
        elevation : array_like
            Elevations to compute velocity coefficient.

        Returns
        -------
        vel_coeff : float or numpy.ndarray

        """

        elevation = np.array(elevation, dtype=np.float)
        elevation_dims = elevation.ndim
        if elevation_dims == 0:
            elevation = elevation[np.newaxis]

        # to hold summed terms
        sigma = np.zeros_like(elevation)

        for ss in self._subsections:
            a_ss = ss.area(elevation)
            k_ss = ss.conveyance(elevation)

            zero_a = a_ss == 0

            if np.any(~zero_a):
                sigma[~zero_a] += (k_ss[~zero_a]**3)/(a_ss[~zero_a]**2)

        area = self.area(elevation)
        k_t = self.conveyance(elevation)

        zero_k_t = k_t == 0
        velocity_coeff = np.zeros_like(elevation)
        velocity_coeff[zero_k_t] = np.nan
        velocity_coeff[~zero_k_t] = area[~zero_k_t]**2 / \
            k_t[~zero_k_t]**3*sigma[~zero_k_t]

        if elevation_dims == 0:
            return float(velocity_coeff)
        else:
            return velocity_coeff

    def vel_dist_factor(self, elevation):
        """Computes the velocity distribution factor (beta) for this
        cross section

        Parameters
        ----------
        elevation : array_like
            Elevations to compute velocity distribution factor

        Returns
        -------
        vel_dist_factor : float or numpy.ndarray

        """

        elevation = np.array(elevation, dtype=np.float)
        elevation_dims = elevation.ndim
        if elevation_dims == 0:
            elevation = elevation[np.newaxis]

        # to hold summed terms
        sigma = np.zeros_like(elevation)

        for ss in self._subsections:
            a_ss = ss.area(elevation)
            k_ss = ss.conveyance(elevation)

            zero_a = a_ss == 0

            if np.any(~zero_a):
                sigma[~zero_a] += (k_ss[~zero_a]**2)/(a_ss[~zero_a])

        # total
        area = self.area(elevation)
        k_t = self.conveyance(elevation)

        # set zero conveyance elements to nan and avoid divide by zero warning
        zero_k_t = k_t == 0
        vel_dist_fact = np.zeros_like(elevation)
        vel_dist_fact[zero_k_t] = np.nan
        vel_dist_fact[~zero_k_t] = area[~zero_k_t] / \
            k_t[~zero_k_t]**2*sigma[~zero_k_t]

        if elevation_dims == 0:
            return float(vel_dist_fact)
        else:
            return vel_dist_fact

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

        if self._wall:
            wall = 'lr'
        else:
            wall = None

        return self._array.perimeter(elevation, wall)
