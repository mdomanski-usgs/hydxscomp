from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Frict:
    """Interface for cross section roughness"""

    @abstractmethod
    def roughness(self, *args):
        """Returns cross section roughness"""

        pass


class TableFrict(Frict):
    """Linearly interpolates Manning's n

    `stage` and `roughness` must be one-dimensional and have two elements
    or more. `stage` must be sorted in ascending order.

    Parameters
    ----------
    stage : array_like
        Water surface elevations
    roughness : array_like
        Manning's n values

    """

    def __init__(self, stage, roughness):

        self._stage = np.array(stage)
        self._roughness = np.array(roughness)

        if self._stage.ndim != 1:
            raise ValueError("stage must be one-dimensional")

        if self._stage.size < 2:
            raise ValueError("stage must at least have two elements")

        if not np.alltrue(np.diff(stage) >= 0):
            raise ValueError("stage must be sorted in ascending order")

        if self._roughness.ndim != 1:
            raise ValueError("roughness must be one-dimensional")

        if self._roughness.size < 2:
            raise ValueError("roughness must at least have two elements")

        if self._stage.size != self._roughness.size:
            raise ValueError("stage and roughness must have the same size")

    def plot(self, ax=None):

        if ax is None:
            ax = plt.axes()

        ax.plot(self._stage, self._roughness)
        ax.set_xlabel('Stage, in ft')
        ax.set_ylabel(r'Manning\'s n')

        return ax

    def roughness(self, stage, *args):
        """Computes Manning's n for a particular elevation of the
        water surface.

        Parameters
        ----------
        stage : array_like
            Water surface elevation

        Returns
        -------
        float or ndarray
            Manning's n

        """

        return np.interp(stage, self._stage, self._roughness)
