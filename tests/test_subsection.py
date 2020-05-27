import unittest
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

    def test_area(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        roughness = 0.035
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, 100)
        self.assertTrue(np.allclose(e, ss.area(e)))

        # double square
        station = [0, 0, 2, 2]
        elevation = [1, 0, 0, 1]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, 100)
        self.assertTrue(np.allclose(2*e, ss.area(e)))

        # triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1]
        elevation = [z, 0, z]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(e**2*np.tan(np.pi/6), ss.area(e)))

        # double triangle
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), ss.area(e)))

        # double triangle w/ shifted elevation
        shift_e = 1.5
        station = [0, 0.5, 1, 1.5, 2]
        elevation = np.array([z, 0, z, 0, z]) + shift_e
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10) + shift_e
        self.assertTrue(np.allclose(2*(e - shift_e) **
                                    2*np.tan(np.pi/6), ss.area(e)))

        # double triangle w/ shifted station
        shift_s = 1.5
        station = np.array([0, 0.5, 1, 1.5, 2]) + shift_s
        elevation = [z, 0, z, 0, z]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), ss.area(e)))

    def test_top_width(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        roughness = 0.035
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, 100)
        tw = np.ones_like(e)
        tw[0] = 0
        self.assertTrue(np.allclose(tw, ss.top_width(e)))

        # double square
        station = [0, 0, 2, 2]
        elevation = [1, 0, 0, 1]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, 100)
        tw = 2*np.ones_like(e)
        tw[0] = 0
        self.assertTrue(np.allclose(tw, ss.top_width(e)))

        # triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1]
        elevation = [z, 0, z]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10)
        tw = 2*e*np.tan(np.pi/6)
        self.assertTrue(np.allclose(tw, ss.top_width(e)))

        # double triangle
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        ss = SubSection(station, elevation, roughness)
        e = np.linspace(0, z, 10)
        tw = 2*2*e*np.tan(np.pi/6)
        self.assertTrue(np.allclose(tw, ss.top_width(e)))


if __name__ == '__main__':
    unittest.main()
