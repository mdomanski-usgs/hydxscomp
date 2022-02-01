import os
import glob
from importlib import util
import unittest

absolute_path, _ = os.path.split(os.path.realpath(__file__))

module_paths = \
    [path for path in glob.glob(os.path.join(
        absolute_path, 'tests', '*.py')) if '__init__.py' not in path]


def load_tests(loader, *args):

    suite = unittest.TestSuite()

    for path in module_paths:
        _, module_file_name = os.path.split(path)
        module_name, _ = os.path.splitext(module_file_name)
        spec = util.spec_from_file_location(module_name, path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)

        tests = loader.loadTestsFromModule(module)

        suite.addTest(tests)

    return suite


if __name__ == '__main__':
    test_loader = unittest.defaultTestLoader
    test_suite = load_tests(test_loader)
    unittest.TextTestRunner().run(test_suite)
