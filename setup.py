
from setuptools import setup, find_packages
from svd.core.version import get_version

VERSION = get_version()

f = open('README.md', 'r')
LONG_DESCRIPTION = f.read()
f.close()

setup(
    name='svd',
    version=VERSION,
    description='Software Vulnerability Detection',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Eduard Pinconschi',
    author_email='up202103584@edu.fe.up.pt',
    url='https://github.com/epicosy/svd',
    license='MIT',
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'svd': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        svd = svd.main:main
    """,
)
