from setuptools import setup

try:
    from sphinx.setup_command import BuildDoc
    sphinx_imported = True
except BaseException:
    sphinx_imported = False

name = 'anchovy'

about = {}
with open('anchovy/__init__.py') as fp:
    exec(fp.read(), about)
release = about['__release__']
version = about['__version__']

dev_status = 'Development Status :: 1 - Planning'

install_requires = ['matplotlib', 'numpy']

setup_kwargs = {
    'name': name,
    'version': release,
    'packages': ['anchovy'],
    'url': 'https://code.usgs.gov/mdomanski/aluminiumanchovy',
    'license': 'License :: Public Domain',
    'author': 'Marian Domanski',
    'author_email': 'mdomanski@usgs.gov',
    'description': 'Hydraulic geometry',
    'classifiers': [
        dev_status,
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Hydrology'
    ],
    'python_requires': '~=3.7',
    'install_requires': install_requires
}

if sphinx_imported:
    cmdclass = {'build_sphinx': BuildDoc}
    docs_source = 'docs/'
    docs_build_dir = 'docs/_build'
    docs_builder = 'html'
    setup_kwargs['command_options'] = {'build_sphinx': {
        'project': ('setup.py', name),
        'version': ('setup.py', version),
        'release': ('setup.py', release),
        'source_dir': ('setup.py', docs_source),
        'build_dir': ('setup.py', docs_build_dir),
        'builder': ('setup.py', docs_builder)}}

setup(**setup_kwargs)
