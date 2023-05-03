from setuptools import setup, find_packages, Extension
# from Cython.Build import cythonize


def readme():
    with open('README.md') as f:
        return f.read()


# cythonize awesomeness
# ext_modules = ["cana/cutils.pyx", "cana/canalization/cboolean_canalization.pyx"]
extensions = [Extension("cana.cutils", ["cana/cutils.c"]), Extension("cana.canalization.cboolean_canalization",["cana/canalization/cboolean_canalization.c"])]

__package__ = 'cana'
__description__ = u'This package implements a series of methods used to study control, canalization and redundancy in Boolean networks.'
__version__ = '0.2.0'

setup(
    name=__package__,
    version=__version__,
    description=__description__,
    long_description=__description__,
    long_description_content_type="text/plain",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords="boolean networks canalization redundancy dynamical systems computational biology",
    url="http://github.com/rionbr/CANA",
    author="Alex Gates & Rion Brattig Correia",
    author_email="rionbr@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={
        'datasets': [
            'cana.datasets/*.txt',
            'cana.datasets/bns/*.cnet',
            'cana.datasets/cell_collective/*.txt'
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'pandas',
        # 'Cython'
    ],
    include_package_data=True,
    zip_safe=False,
    # ext_modules=cythonize(ext_modules, include_path=[''], compiler_directives={'language_level': '3'})  # cython awesomeness
    ext_modules=extensions,
)
