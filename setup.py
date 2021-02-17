import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pacoh_nn",
    version="0.0.1",
    author="Jonas Rothfuss, Martin Josifoski",
    author_email="jonas.rothfuss@gmail.com",
    description="Meta-Learning Bayesian Neural Network Priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'pacoh_nn': 'pacoh_nn'},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tensorflow>=2.1.0',
        'tensorflow-probability>=0.9.0',
        'matplotlib',
        'pyYAML',
    ],
)