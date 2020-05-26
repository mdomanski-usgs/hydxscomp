from unittest import TestCase

import numpy as np

from anchovy.subsection import SubSection


class TestSubSection(TestCase):

    def test_init(self):

        station = [0, 1, 2, 3]
        elevation = [1, 0, 0, 1]
        roughness = 0.035

        # successful init
        ss = SubSection(station, elevation, roughness)
        self.assertIsInstance(ss, SubSection)

        # non-ascending station
        args = ([1, 0, 2, 3], elevation, roughness)
        self.assertRaisesRegex(
            ValueError, "station must be in ascending order",
            SubSection, *args)

        # non-finite station
        args = ([0, np.nan, 2, 3], elevation, roughness)
        self.assertRaisesRegex(
            ValueError, "station must be finite",
            SubSection, *args)

        # non-finite elevation
        args = (station, [1, np.inf, 0, 1], roughness)
        self.assertRaisesRegex(
            ValueError, "elevation must be finite",
            SubSection, *args)

        # non-finite roughness
        args = (station, elevation, np.inf)
        self.assertRaisesRegex(
            ValueError, "roughness must be finite",
            SubSection, *args)

        # 2-d station
        args = ([[0, 1, 2, 3]], elevation, roughness)
        self.assertRaisesRegex(
            ValueError, "station must be one dimensional", SubSection, *args)

        # 2-d elevation
        args = (station, [[1, 0, 0, 1]], roughness)
        self.assertRaisesRegex(
            ValueError, "elevation must be one dimensional", SubSection, *args)

        # different-sized station, elevation
        args = ([0, 1, 2], elevation, roughness)
        self.assertRaisesRegex(
            ValueError, "station and elevation must have the same size",
            SubSection, *args)
        args = (station, [1, 0, 1], roughness)
        self.assertRaisesRegex(
            ValueError, "station and elevation must have the same size",
            SubSection, *args)
