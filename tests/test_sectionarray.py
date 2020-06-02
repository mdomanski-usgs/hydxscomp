import unittest
from unittest import TestCase

import numpy as np

from anchovy.crosssection import SectionArray


class TestSectionArray(TestCase):

    def test_init(self):
        station = [0, 1, 2, 3]
        elevation = [1, 0, 0, 1]

        # successful init
        section_array = SectionArray(station, elevation)
        self.assertIsInstance(section_array, SectionArray)

        # non-ascending station
        args = ([1, 0, 2, 3], elevation)
        self.assertRaisesRegex(
            ValueError, "station must be in ascending order",
            SectionArray, *args)

        # non-finite station
        args = ([0, np.nan, 2, 3], elevation)
        self.assertRaisesRegex(
            ValueError, "station must be finite",
            SectionArray, *args)

        # non-finite elevation
        args = (station, [1, np.inf, 0, 1])
        self.assertRaisesRegex(
            ValueError, "elevation must be finite",
            SectionArray, *args)

        # 2-d station
        args = ([[0, 1, 2, 3]], elevation)
        self.assertRaisesRegex(
            ValueError, "station must be one dimensional", SectionArray, *args)

        # 2-d elevation
        args = (station, [[1, 0, 0, 1]])
        self.assertRaisesRegex(
            ValueError, "elevation must be one dimensional",
            SectionArray, *args)

        # different-sized station, elevation
        args = ([0, 1, 2], elevation)
        self.assertRaisesRegex(
            ValueError, "station and elevation must have the same size",
            SectionArray, *args)
        args = (station, [1, 0, 1])
        self.assertRaisesRegex(
            ValueError, "station and elevation must have the same size",
            SectionArray, *args)

    def test_area(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, 100)
        self.assertTrue(np.allclose(e, sa.area(e)))

        # double square
        station = [0, 0, 2, 2]
        elevation = [1, 0, 0, 1]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, 100)
        self.assertTrue(np.allclose(2*e, sa.area(e)))

        # triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1]
        elevation = [z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(e**2*np.tan(np.pi/6), sa.area(e)))

        # double triangle
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), sa.area(e)))

        # double triangle w/ shifted elevation
        shift_e = 1.5
        station = [0, 0.5, 1, 1.5, 2]
        elevation = np.array([z, 0, z, 0, z]) + shift_e
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10) + shift_e
        self.assertTrue(np.allclose(2*(e - shift_e) **
                                    2*np.tan(np.pi/6), sa.area(e)))

        # double triangle w/ shifted station
        shift_s = 1.5
        station = np.array([0, 0.5, 1, 1.5, 2]) + shift_s
        elevation = [z, 0, z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), sa.area(e)))

    def test_perimeter(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, 1, 10)
        perimeter = 1 + 2*e
        perimeter[0] = 0
        self.assertTrue(np.allclose(perimeter, sa.perimeter(e)))

        # double triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        perimeter = 2*2*e/np.cos(np.pi/6)
        self.assertTrue(np.allclose(perimeter, sa.perimeter(e)))

    def test_top_width(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, 100)
        tw = np.ones_like(e)
        tw[0] = 0
        self.assertTrue(np.allclose(tw, sa.top_width(e)))

        # double square
        station = [0, 0, 2, 2]
        elevation = [1, 0, 0, 1]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, 100)
        tw = 2*np.ones_like(e)
        tw[0] = 0
        self.assertTrue(np.allclose(tw, sa.top_width(e)))

        # triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1]
        elevation = [z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        tw = 2*e*np.tan(np.pi/6)
        self.assertTrue(np.allclose(tw, sa.top_width(e)))

        # double triangle
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        sa = SectionArray(station, elevation)
        e = np.linspace(0, z, 10)
        tw = 2*2*e*np.tan(np.pi/6)
        self.assertTrue(np.allclose(tw, sa.top_width(e)))


if __name__ == '__main__':
    unittest.main()
