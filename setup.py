"""
Setup configuration for EDA Suite.

Installation script for the package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eda-suite",
    version="1.0.0",
    author="EDA Suite Contributors",
    author_email="contact@eda-suite.org",
    description="Comprehensive Exploratory Data Analysis Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="eda exploratory-data-analysis statistics machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis/issues",
        "Source": "https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis",
    },
)
