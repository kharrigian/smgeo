from setuptools import setup

## README
with open("README.md", "r") as fh:
    long_description = fh.read()

## Requirements
with open("requirements.txt", "r") as r:
    requirements = [i.strip() for i in r.readlines()]

## Run Setup
setup(
    name="smgeo",
    version="0.0.1",
    author="Keith Harrigian",
    author_email="kharrigian@jhu.edu",
    description="Geolocation Inference for Social Media",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kharrigian/smgeo",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)