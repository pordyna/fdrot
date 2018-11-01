from distutils.core import setup
from Cython.Build import cythonize

extensions = [Extension('multiplestepsoverx', ['multiplestepsoverx.pyx'])]
setup(
    ext_modules=cythonize(extensions, gdb_debug=True)
)
