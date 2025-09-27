#!/usr/bin/env python3
"""
Setup script for the DanielGarza microbiome modeling packages.
"""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="daniel-garza-microbiome",
    version="0.1.0",
    author="Daniel Rios Garza",
    author_email="daniel.rios.garza@example.com",
    description="Kinetic modeling and reinforcement learning for microbial community dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielriosgarza/AiSchool",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.8",
        "plotly>=5.0",
        "gym>=0.21",
        "stable-baselines3>=2.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5",
        "pandas>=1.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kinetic-model=kinetic_model.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kinetic_model": ["py.typed"],
        "rl": ["examples/configs/*.yaml", "docs/*.md"],
    },
    zip_safe=False,
)
