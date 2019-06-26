import setuptools

__packagename__ = 'calibr8'

def get_version():
    import os, re
    VERSIONFILE = os.path.join(__packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Package for fitting of multiple replicates of a bioprocess dataset.',
        url='https://gitlab.com/diginbio-fzj/murefi',
        author='Laura Marie Helleckes',
        author_email='laurahelleckes@gmail.com',
        copyright='(c) 2019 Forschungszentrum Jülich GmbH',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Intended Audience :: Developers'
        ],
        install_requires=[
            'numpy',
            'scipy'
        ]
)