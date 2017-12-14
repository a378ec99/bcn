from setuptools import setup, find_packages

setup(
    name='BCN',
    version='0.1.0',
    description='Blind Compressive Normalization',
    author='Sebastian Ohse',
    author_email='sebastian.ohse@mailbox.org',
    url='https://github.com/a378ec99/bcn',
    license='LICENSE.txt',
    long_description=open('README.txt').read(),
    keywords=('bias recovery, blind, manifold optimization'),
    packages=find_packages(exclude=['tests']),
    install_requires=['anaconda >= 5.0.1', 'pymanopt >= 0.2.3', 'mpi4py >= 3.0.0'],
)
