Development 
============

This package is still in alpha development. Please note that it might contain errors.

Notes
------

Here are some TODOs:

* Parallelize control methods;
* Parallelize canalization methods;
* There are more code on ``iu.github.edu`` that needs conversion;
* Generating ensembles of dynamics from structural motifs;
* Better document networks from Cell Collective;
* Expand on easy-to-replicate tutorials;

Developing
-----------

Pull requests are welcome :)
Please get in contact with one of us beforehand: ``rionbr(at)gmail(dot)com`` or ``ajgates(at)indiana(dot)edu``.


Changelog
-----------

v0.0.2
	- NetworkX 2.1 compatibility
	- New tutorials
	- Derrida curve bugfix
v0.0.1
	- Code ported to public package.
	- Documentation and docstring added.
	- Control (FVS,MDS,CSTG) methods ported.
	- Canalization methods implemented.
	- BooleanNetwork() ported and BooleanNode() developed.
	- First private implementation (Alex).

Tests
------
Run ``nosetests -v`` to perform tests and diagnoses on functions.