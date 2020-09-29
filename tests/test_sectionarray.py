import unittest
from unittest import TestCase

import numpy as np

from anchovy.sectionarray import SectionArray


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

    def test_active_elev(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        active_elev = 0.5
        sa = SectionArray(station, elevation, active_elev=active_elev)
        e = np.linspace(0, 1)
        area = e.copy()
        area[e < active_elev] = 0

        self.assertTrue(np.allclose(area, sa.area(e)))

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

    def test_split(self):

        station = [0, 0, 300, 300, 600, 600, 900, 900]
        elevation = [9, 3, 3, 0, 0, 3, 3, 9]

        sect_stat = [300, 600]

        start_station = [0, 300, 600]
        end_station = [300, 600, 900]

        start_elevation = [9, 3, 3]
        end_elevation = [3, 3, 9]

        array = SectionArray(station, elevation)

        stages = np.linspace(0, 9)

        perimeter_sum = 0
        area_sum = 0
        tw_sum = 0

        for i, sa in enumerate(array.split(sect_stat)):
            s, e = sa.coordinates()
            self.assertEqual(start_station[i], s[0])
            self.assertEqual(end_station[i], s[-1])
            self.assertEqual(start_elevation[i], e[0])
            self.assertEqual(end_elevation[i], e[-1])
            perimeter_sum += sa.perimeter(stages)
            area_sum += sa.area(stages)
            tw_sum += sa.top_width(stages)

        self.assertTrue(np.allclose(perimeter_sum, array.perimeter(stages)))
        self.assertTrue(np.allclose(area_sum, array.area(stages)))
        self.assertTrue(np.allclose(tw_sum, array.top_width(stages)))

    def test_split_active(self):

        # double triangle w/ two subsections
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        sect_stat = 1

        sa = SectionArray(station, elevation)

        arrays = sa.split(sect_stat, [-np.inf, z])

        e = np.linspace(0, z, 10)
        area = e[:-1]**2*np.tan(np.pi/6)

        self.assertTrue(np.allclose(area, arrays[0].area(e[:-1])))
        self.assertTrue(np.allclose(
            np.zeros_like(area), arrays[1].area(e[:-1])))

        e = np.linspace(z, 2*z, 10)
        self.assertTrue(np.allclose(
            arrays[0].area(e), arrays[1].area(e)))

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
