import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyro_cg", # Replace with your own username
    version="0.0.2",
    author="Craig Bakker",
    author_email="craig.bakker@pnnl.gov",
    description="A package for creating causal graphs in pyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbakker2/pyro_cg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)