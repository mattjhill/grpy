from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
 
setup(
  name = "grpy",
  version = "0.1",
  author = "Matthew Hill",
  author_email = "matthew.hill@yale.edu",
  packages=["grpy"],
  ext_modules=[ 
    Extension("grpy._grp", 
              sources=["grpy/_grp.pyx"], # Note, you can link against a c++ library instead of including the source
              language="c++"),
    ],
  include_dirs=[np.get_include(), "ESS/"],
  cmdclass = {'build_ext': build_ext},
  description = "wrapper for Generalized Rybicki Press algorithm"
)