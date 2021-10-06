import unittest
from unittest import TestCase

from hydxscomp.frict import TableFrict


class TestTableFrict(TestCase):

    def test_init(self):

        elevation = [-100, 100]
        roughness = [0.011, 0.035]

        # successful init
        frict = TableFrict(elevation, roughness)
        self.assertIsInstance(frict, TableFrict)
