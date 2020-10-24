import setuptools
import os
import pathlib
import re

__packagename__ = 'calibr8'
ROOT = pathlib.Path(__file__).parent


def package_files(directory):
    assert os.path.exists(directory)
    fp_typed = pathlib.Path(ROOT, __packagename__, 'py.typed')
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


setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version=__version__,
    description='Toolbox for non-linear calibration and error modeling.',
    long_description=open(pathlib.Path(ROOT, 'README.md')).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/michaelosthege/calibr8',
    author='Laura Marie Helleckes, Michael Osthege',
    author_email='l.helleckes@fz-juelich.de, m.osthege@fz-juelich.de',
    license='AGPLv3',
    classifiers= [
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=[
        open(pathlib.Path(ROOT, 'requirements.txt')).readlines()
    ],
    package_data={
        'calibr8': package_files(str(pathlib.Path(pathlib.Path(__file__).parent, 'calibr8').absolute()))
    },
    include_package_data=True,
    python_requires='>=3.6'
)
