from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name = "cnn_anime",
    version = "1.0",
    packages = find_packages(),
    cmdclass={'build_ext': build_ext},
    setup_requires=[
        'numpy',
        'six'
    ],
    install_requires=[
        'numpy',
        'six',
        'Pillow',
        'scikit-image',
        'Theano'
    ],
    dependency_links=[
        'git+git://github.com/Theano/Theano.git#egg=Theano'
    ]
)
