from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('rotation', ['rotation.pyx'])]
setup(
    ext_modules=cythonize(extensions, gdb_debug=False)
)
