import os

from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="KNN_CUDA",
    version="0.2",
    description=(
        "GPU kNN for PyTorch (community re-upload of unlimblue/KNN_CUDA). "
        "Ships C++/CUDA sources that are JIT-compiled on first import."
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/altaykacan/KNN_CUDA_reborn",
    license="MIT (packaging only -- see DISCLAIMER and source headers)",
    packages=find_packages(),
    # The wheel is pure-Python (py3-none-any). The actual CUDA kernels live in
    # knn_cuda/csrc and are compiled at runtime with torch.utils.cpp_extension,
    # so the sources must be shipped inside the wheel as package data.
    package_data={
        "knn_cuda": [
            "csrc/cuda/*.cpp",
            "csrc/cuda/*.cu",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "torch",
        "ninja",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
