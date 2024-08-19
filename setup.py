import os
import platform
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc, get_python_lib

import pybind11
import numpy as np

is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"
is_nvcc_available = os.system("which nvcc > /dev/null 2>&1") == 0
enable_openmp = not is_macos
enable_cuda = is_linux and is_nvcc_available


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")

        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                if is_nvcc_available:
                    self.build_cuda_extension(ext)
                else:
                    self.build_gcc_extension(ext)
            else:
                super().build_extension(ext)

    def build_cuda_extension(self, ext):
        # Compile CUDA source files
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_cuda(source)

        # Compile non-CUDA source files
        objects = []
        for source in ext.sources:
            if not source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ],
                )
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "cupy_starter.o")],
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def build_gcc_extension(self, ext):
        # Check if compiling on macOS
        objects = []
        for source in ext.sources:
            if source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_preargs=["-x", "c++"],
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ]
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            else:
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ]
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=[lib for lib in ext.libraries if not lib.startswith("cu")],
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def compile_cuda(self, source):
        # Compile CUDA source file using nvcc
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        include_dirs = self.compiler.include_dirs + ext.include_dirs
        include_dirs = " ".join(f"-I{dir}" for dir in include_dirs)
        output_file = os.path.join(output_dir, "cupy_starter.o")

        # Let's try inferring the compute capability from the GPU
        arch_code = "90"
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            device = cuda.Device(0)  # Get the default device
            major, minor = device.compute_capability()
            arch_code = f"{major}{minor}"
        except ImportError:
            pass

        cmd = (
            f"nvcc -c {source} -o {output_file} -std=c++17 "
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code} "
            f"-Xcompiler -fPIC {include_dirs} -O3 -g"
        )
        if os.system(cmd) != 0:
            raise RuntimeError(f"nvcc compilation of {source} failed")


__version__ = "0.0.1"

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

# Get Python library path dynamically
python_lib_dir = get_python_lib(standard_lib=True)
python_lib_name = os.path.basename(python_lib_dir).replace(".so", "")

ext_modules = [
    Extension(
        "cupy_starter",
        ["cupy_starter.cu"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),
            "cccl/cub/",
            "cccl/libcudacxx/include",
            "cccl/thrust/",
            "/usr/local/cuda/include/",
            "/usr/include/cuda/",
        ],
        library_dirs=[
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/wsl/lib",
            python_lib_dir,
        ],
        #
        libraries=[python_lib_name.replace(".a", "")]
        + (["cudart", "cuda", "cublas"] if enable_cuda else [])
        + (["gomp"] if enable_openmp else []),
        #
        extra_link_args=[f"-Wl,-rpath,{python_lib_dir}"]
        + (["-fopenmp"] if enable_openmp else []),
        language="c++",
    ),
]

setup(
    name="cupy_starter",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/cuda-python-starter-kit",
    description="Starter Kit project for CUDA- and OpenMP-accelerated Python projects.",
    long_description=long_description,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
