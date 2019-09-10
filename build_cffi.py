from cffi import FFI
import os

ffi = FFI()
LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, "RIMEz", "dftpack_wrappers")
include_dirs = [CLOC]

library_dirs = []
for k, v in os.environ.items():
    if "inc" in k.lower():
        include_dirs += [v]
    elif "lib" in k.lower():
        library_dirs += [v]


# This is the overall C code.
ffi.set_source(
    "RIMEz.dfitpack_wrappers",  # Name/Location of shared library module
    """
    #define LOG_LEVEL {log_level}

    #include "GenerateICs.c"
    """.format(
        log_level=log_level
    ),
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["m", "gsl", "gslcblas", "fftw3f_omp", "fftw3f"],
    extra_compile_args=extra_compile_args,
    extra_link_args=["-fopenmp"],
)

# This is the Header file
with open(os.path.join(CLOC, "21cmFAST.h")) as f:
    ffi.cdef(f.read())

with open(os.path.join(CLOC, "Globals.h")) as f:
    ffi.cdef(f.read())

if __name__ == "__main__":
    ffi.compile()
