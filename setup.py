import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bel2scm", # Replace with your own username
    version="0.0.2",
    author="Craig Bakker",
    author_email="craig.bakker@pnnl.gov",
    description="A package for creating causal graphs in pyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/COVID-19-Causal-Reasoning/bel2scm",
    packages=setuptools.find_packages(),
    install_requires=[
    	'numpy',
    	'torch',
    	'pyro',
    	'networkx',
    	'scipy',
    	'json',
    	'pybel'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)