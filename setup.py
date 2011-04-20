from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("cybvh",
                       ["cybvh.pyx"],
                       libraries=['m']),
             ]

setup(name='mesh',
      version='0.01',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
