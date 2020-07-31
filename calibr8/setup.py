import setuptools
import os
import pathlib
import re

__packagename__ = 'calibr8'


def package_files(directory):
    assert os.path.exists(directory)
    fp_typed = pathlib.Path(pathlib.Path(__file__).parent, __packagename__, 'py.typed')
    fp_typed.touch()
    paths = [str(fp_typed.absolute())]
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def get_version():
    VERSIONFILE = pathlib.Path(pathlib.Path(__file__).parent, __packagename__, 'core.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(),
        version=__version__,
        description='Toolbox for non-linear calibration and error modeling.',
        url='https://jugit.fz-juelich.de/ibg-1/micropro/calibr8',
        author='Laura Marie Helleckes, Michael Osthege',
        author_email='l.helleckes@fz-juelich.de, m.osthege@fz-juelich.de',
        license='(c) 2020 Forschungszentrum Jülich GmbH',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.7',
            'Intended Audience :: Developers'
        ],
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'fastprogress',
        ],
        package_data={
            'calibr8': package_files(str(pathlib.Path(pathlib.Path(__file__).parent, 'calibr8').absolute()))
        },
        include_package_data=True
)