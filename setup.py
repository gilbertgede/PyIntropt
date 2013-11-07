from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import os
import sys
import subprocess

args = sys.argv[1:]


if "cleanall" in args:
    print("Deleting cython files...")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf pyintropt/*.c", shell=True, executable="/bin/bash")
    #subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
    sys.argv[1] = "clean"

if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

setup(
    name = 'pyintropt',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include(), '\opt\local\include'],
    ext_modules = [
        Extension("pyintropt.ss_spqr",
              sources=["pyintropt/ss_spqr.pyx"],
              libraries=["SuiteSparse", "m"],
              extra_compile_args=["-fopenmp", "-O3"],
              )
    ]
)



