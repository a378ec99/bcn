from setuptools import setup, find_packages

setup(
      name = 'bcn',
      version = '0.1.0',
      description = 'Blind Compressive Normalization',
      author = 'Sebastian Ohse',
      author_email = 'sebastian.ohse@mailbox.org',
      url = 'https://github.com/a378ec99/bcn',
      license = 'LICENSE',
      long_description = open('README.md').read(),
      keywords = ('bias', 'normalization', 'blind', 'manifold optimization', 'compressed sensing', 'high-dimensional'),
      packages = find_packages(exclude=['tests']),
      install_requires = ['scipy >= 0.19.0', 'numpy >= 1.11.3', 'sklearn >= 0.18.1', 'pymanopt >= 0.2.3']
      extras_require = {
                        'visualization': ['seaborn >= 0.7.1', 'matplotlib >= 1.5.3'],
                        'parallel': ['mpi4py >= 3.0.0']
                       }
)
