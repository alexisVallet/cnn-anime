from setuptools import setup, find_packages

setup(
    name = "cnn_anime",
    version = "1.0",
    packages = find_packages(),
    dependency_links=['git+git://github.com/Theano/Theano.git']
)
