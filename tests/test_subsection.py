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


if __name__ == '__main__':
    unittest.main()
