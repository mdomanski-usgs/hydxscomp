import unittest
from unittest import TestCase

import numpy as np

from anchovy.crosssection import SubSection
from anchovy.crosssection import SectionArray


class TestSubSection(TestCase):

    def test_init(self):

        station = [0, 1, 2, 3]
        elevation = [1, 0, 0, 1]
        roughness = 0.035

        # successful init
        section_array = SectionArray(station, elevation)
        ss = SubSection(section_array, roughness)
        self.assertIsInstance(ss, SubSection)

        # non-finite roughness
        section_array = SectionArray(station, elevation)
        args = (section_array, np.inf)
        self.assertRaisesRegex(
            ValueError, "roughness must be finite",
            SubSection, *args)

    def test_perimeter(self):

        station = [0, 1]
        elevation = [0, 0]
        roughness = 0.035

        section_array = SectionArray(station, elevation)
        ss = SubSection(section_array, roughness)
        self.assertEqual(ss.wetted_perimeter(1), 1)

        ss_l_wall = SubSection(section_array, roughness, wall='l')
        self.assertEqual(ss_l_wall.wetted_perimeter(1), 2)

        ss_r_wall = SubSection(section_array, roughness, wall='r')
        self.assertEqual(ss_r_wall.wetted_perimeter(1), 2)

        ss_lr_wall = SubSection(section_array, roughness, wall='lr')
        self.assertEqual(ss_lr_wall.wetted_perimeter(1), 3)


if __name__ == '__main__':
    unittest.main()
