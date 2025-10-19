"""
Setup configuration for quantum-consciousness research framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-consciousness",
    version="0.1.0",
    author="OmniSphere",
    author_email="research@omnisphere.org",
    description="Multi-scale quantum consciousness research framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmanthe37/quantum-consciousness",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "mne>=1.0.0",  # For EEG/MEG processing
        "nilearn>=0.8.0",  # For fMRI processing
        "pymc>=4.0.0",  # For Bayesian modeling
        "torch>=1.10.0",  # For neural networks
        "statsmodels>=0.13.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "h5py>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-consciousness=quantum_consciousness.cli:main",
        ],
    },
)