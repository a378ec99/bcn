from setuptools import setup, find_packages

setup(
    name='BCN',
    version='0.1.0',
    description='Blind Compressive Normalization',
    author='Sebastian Ohse',
    author_email='sebastian.ohse@mailbox.org',
    url='https://github.com/a378ec99/bcn',
    license='LICENSE.txt',
    long_description=open('README.me').read(),
    keywords=('bias', 'normalization', 'blind', 'manifold optimization', 'compressed sensing', 'high-dimensional'),
    packages=find_packages(exclude=['tests']),
    install_requires=['scipy >= 0.19.0', 'numpy >= 1.11.3', 'sklearn >= 0.18.1', 'pymanopt >= 0.2.3', 'mpi4py >= 3.0.0', 'matplotlib >= 1.5.3', 'seaborn >= 0.7.1', 'mpi4py >= 3.0.0']
)

