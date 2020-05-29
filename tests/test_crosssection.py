import unittest
from unittest import TestCase

import numpy as np

from anchovy.crosssection import CrossSection


class TestCrossSection(TestCase):

    def test_area(self):

        # unit square w/ 3 subsections
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        roughness = [0.035, 0.010, 0.035]
        sect_stat = [0.25, 0.75]
        xs = CrossSection(station, elevation, roughness, sect_stat)
        e = np.linspace(0, 100)
        self.assertTrue(np.allclose(e, xs.area(e)))

        # double triangle w/ two subsections
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        roughness = [0.035, 0.035]
        sect_stat = 1
        xs = CrossSection(station, elevation, roughness, sect_stat)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), xs.area(e)))

        # double triangle w/ three subsections
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        roughness = [0.035, 0.035, 0.035]
        sect_stat = [1, 1.75]
        xs = CrossSection(station, elevation, roughness, sect_stat)
        e = np.linspace(0, z, 10)
        self.assertTrue(np.allclose(2*e**2*np.tan(np.pi/6), xs.area(e)))

    def test_top_width(self):

        # unit square
        station = [0, 0, 1, 1]
        elevation = [1, 0, 0, 1]
        roughness = [0.035, 0.010, 0.035]
        sect_stat = [0.25, 0.75]
        xs = CrossSection(station, elevation, roughness, sect_stat)
        e = np.linspace(0, 100)
        tw = np.ones_like(e)
        tw[0] = 0
        self.assertTrue(np.allclose(tw, xs.top_width(e)))

        # double triangle
        z = np.cos(np.arcsin(0.5))
        station = [0, 0.5, 1, 1.5, 2]
        elevation = [z, 0, z, 0, z]
        roughness = [0.035, 0.035]
        sect_stat = 1
        xs = CrossSection(station, elevation, roughness, sect_stat)
        e = np.linspace(0, z, 10)
        tw = 2*2*e*np.tan(np.pi/6)
        self.assertTrue(np.allclose(tw, xs.top_width(e)))


if __name__ == '__main__':
    unittest.main()