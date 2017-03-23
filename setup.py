from setuptools import setup
from boolnets import *

def readme():
	with open('README.md') as f:
		return f.read()

setup(
	name=__package__.title(),
	version=__version__,
	description="CANalization: Control & Redundancy in Boolean Networks",
	long_description="",
	classifiers=[
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Information Analysis',
	],
	keywords="boolean networks canalization redundancy dynamical systems computational biology",
	url="http://github.com/rionbr/CAN",
	author="Alex Gates & Rion Brattig Correia",
	author_email="rionbr@gmail.com",
	license="MIT",
	packages=['can'],
	install_requires=[
		'numpy',
		'scipy',
		'networkx',
		'pandas'
	],
	include_package_data=True,
	zip_safe=False,
	)