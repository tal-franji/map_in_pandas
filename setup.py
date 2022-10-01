from setuptools import find_packages
from setuptools import setup

setup(
    name="franji-mapinpandas",
    version="0.5",
    packages=find_packages(),
    url="https://github.com/tal-franji/map_in_pandas",
    author="tal-franji",
    author_email="tal.franji@gmail.com",
    description="Easy python wrapper for Spark mapInPandas, applyInPandas",
    package_data={"doc": ["*.ipynb"]},
    keywords='spark pyspark',
)
