import unittest
from unittest import TestCase

from hydxscomp.crosssection import SubSection
from hydxscomp.crosssection import SectionArray
from hydxscomp.frict import TableFrict


class TestSubSection(TestCase):

    def test_init(self):

        station = [0, 1, 2, 3]
        elevation = [1, 0, 0, 1]

        frict = TableFrict([0, 1], [0.035, 0.035])

        # successful init
        section_array = SectionArray(station, elevation)
        ss = SubSection(section_array, frict)
        self.assertIsInstance(ss, SubSection)

    def test_perimeter(self):

        station = [0, 1]
        elevation = [0, 0]

        frict = TableFrict([0, 1], [0.035, 0.035])

        section_array = SectionArray(station, elevation)
        ss = SubSection(section_array, frict)
        self.assertEqual(ss.wetted_perimeter(1), 1)

        ss_l_wall = SubSection(section_array, frict, wall='l')
        self.assertEqual(ss_l_wall.wetted_perimeter(1), 2)

        ss_r_wall = SubSection(section_array, frict, wall='r')
        self.assertEqual(ss_r_wall.wetted_perimeter(1), 2)

        ss_lr_wall = SubSection(section_array, frict, wall='lr')
        self.assertEqual(ss_lr_wall.wetted_perimeter(1), 3)


if __name__ == '__main__':
    unittest.main()
