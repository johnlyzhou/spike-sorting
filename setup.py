import os
from pathlib import Path
from setuptools import find_packages, setup


setup(
    name='src',
    packages=find_packages(),
)

# Set up directories
repo_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = Path(f"{repo_dir}/data")
raw_data_dir = Path(f"{data_dir}/raw")
process_data_dir = Path(f"{data_dir}/processed")
figures_dir = Path(f"{repo_dir}/figures")
experiments_dir = Path(f"{repo_dir}/experiments")

data_dir.mkdir(parents=True, exist_ok=True)
raw_data_dir.mkdir(parents=True, exist_ok=True)
process_data_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
experiments_dir.mkdir(parents=True, exist_ok=True)
