from setuptools import setup, find_packages
import cana
from cana import __package__, __title__, __description__, __version__

def readme():
	with open('README.md') as f:
		return f.read()

setup(
	name=__package__,
	version=__version__,
	description=__description__,
	long_description=__description__,
	classifiers=[
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
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
		'pandas'
	],
	include_package_data=True,
	zip_safe=False,
	)