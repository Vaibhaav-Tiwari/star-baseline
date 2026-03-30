from setuptools import setup, find_packages

setup(
    name="star-spatial",
    version="0.1.0",
    description="StaR: Stability-Aware Representation Learning for Spatial Domain Identification",
    author="Vaibhaav Tiwari",
    author_email="phylolver@gmail.com",
    url="https://github.com/Vaibhaav-Tiwari/star-baseline",
    license="Apache 2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "realdata": ["scanpy>=1.9.0", "squidpy>=1.3.0", "anndata>=0.10.0"],
        "torch":    ["torch>=2.0.0", "torch-geometric>=2.3.0"],
        "dev":      ["pytest>=7.0.0", "wandb>=0.15.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
