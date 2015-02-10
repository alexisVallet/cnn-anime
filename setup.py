from setuptools import setup, find_packages

setup(
    name = "cnn_anime",
    version = "1.0",
    packages = find_packages(),
    setup_requires=[
        'six'
    ],
    install_requires=[
        'numpy',
        'Pillow',
        'scikit-image',
        'Theano'
    ],
    dependency_links=[
        'git+git://github.com/Theano/Theano.git#egg=Theano'
    ]
)
