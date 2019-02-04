from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [Extension('fdrot.Kernel2D', ['fdrot/Kernel2D.pyx'])]
                         # ,define_macros=[('CYTHON_TRACE', '1')])]
setup(name='Fdrot',
      ext_modules=cythonize(extensions,
                            annotate=True, gdb_debug=False,
                            compiler_directives={'embedsignature': True, 'language_level': 3}),
      install_requires=['numpy>=1.15', 'scipy', 'cython>=0.29'],
      packages=find_packages(), version='0.0.3')

