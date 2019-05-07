from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [Extension('fdrot.Kernel2D', ['fdrot/Kernel2D.pyx']),
              Extension('fdrot.test_wraps',
                        ['tests/Kernel2D_testing_wraps.pyx'])]
                         # ,define_macros=[('CYTHON_TRACE', '1')])]
setup(name='fdrot',
      ext_modules=cythonize(extensions,
                            annotate=True, gdb_debug=False,
                            compiler_directives={'embedsignature': True, 'language_level': 3, 'binding': False}),
      install_requires=['numpy>=1.15', 'scipy', 'cython>=0.29', 'numba', 'pytest>=3.9'],
      packages=find_packages(), version='0.0.3')

