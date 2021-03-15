CANAlization: Control & Redundancy in Boolean Networks
=======================================================

This package implements a series of methods used to study control, canalization and redundancy in Boolean Networks.

Installation:
-------------

** Latest development release on GitHub **

Pull and install the code directly from the github [project page](https://github.com/rionbr/CANA).

```
    pip install git+git://github.com/rionbr/CANA
```

** Latest PyPI stable release **

This package is available in pypi. Just run the following command on terminal to install.

```
    $ pip install cana
```


Docs:
-------

The full documentation can be found at: [rionbr.github.io/CANA](https://rionbr.github.io/CANA)


Papers:
---------

- A.J. Gates, R.B. Correia, X. Wang, L.M. Rocha [2021]. "[The effective graph reveals redundancy, canalization, and control pathways in biochemical regulation and signaling](https://doi.org/10.1073/pnas.2022598118)". *Proceedings of the National Academy of Sciences (PNAS)*. 118(**12**). doi: 10.1073/pnas.20225981186

- R.B. Correia, A.J. Gates, X. Wang, L.M. Rocha [2018]. "[CANA: A python package for quantifying control and canalization in Boolean Networks](https://www.informatics.indiana.edu/rocha/publications/FSB18.php)". *Frontiers in Physiology*. **9**: 1046. doi: 10.3389/fphys.2018.01046

- A. Gates and L.M. Rocha. [2016] "[Control of complex networks requires both structure and dynamics.](http://www.informatics.indiana.edu/rocha/publications/NSR16.php)" *Scientific Reports* **6**, 24456. doi: 10.1038/srep24456.

- A. Gates and L.M. Rocha [2014]. "[Structure and dynamics affect the controllability of complex systems: a Preliminary Study](http://www.informatics.indiana.edu/rocha/publications/alife14a.html)". *Artificial Life 14: Proceedings of the Fourteenth International Conference on the Synthesis and Simulation of Living Systems*: 429-430, MIT Press.

- M. Marques-Pita and L.M. Rocha [2013]. "[Canalization and control in automata networks: body segmentation in Drosophila Melanogaster](http://informatics.indiana.edu/rocha/publications/plos2012.html)". *PLoS ONE*, **8**(3): e55946. doi:10.1371/journal.pone.0055946.


Credits:
---------

``CANA`` was originally written by Rion Brattig Correia and Alexander Gates, and has been developed
with the help of many others. Thanks to everyone who has improved ``CANA`` by contributing code, bug reports (and fixes), documentation, and input on design, and features.


**Original Authors**

- [Rion Brattig Correia](http://alexandergates.net/), github: [rionbr](https://github.com/rionbr)
- [Alexander Gates](https://alexandergates.net/), github: [ajgates42](https://github.com/ajgates42)


**Contributors**

Optionally, add your desired name and include a few relevant links. The order
is an attempt at historical ordering.

- [Xuan Wang](https://www.wangxuan.name), github: [xuan-w](https://github.com/xuan-w)
- Thomas Parmer, github: [tjparmer](https://github.com/tjparmer)
- Etienne Nzabarushimana
- Luis M. Rocha


Support
-------

Those who have contributed to ``CANA`` have received support throughout the years from a variety of sources.  We list them below.
If you have provided support to ``CANA`` and a support acknowledgment does not appear below, please help us remedy the situation, and similarly, please let us know if you'd like something modified or corrected.

**Research Groups**

``CANA`` was developed with support from the following:

- [CASCI](https://homes.luddy.indiana.edu/rocha/casci.php), Indiana University, Bloomington, IN; PI: Luis M. Rocha
- [CAPES Foundation](https://www.gov.br/capes/pt-br), Ministry of Education of Brazil, Brasília, Brazil; Rion B. Correia.


Development
-----------
Pull requests are welcome :) Please get in touch with one us beforehand: `rionbr(at)gmail(dot)com` or `ajgates42(at)gmail(dot)com`.

** TODOs**

- Parallelize control methods;
- Parallelize canalization methods;
- Generating ensembles of dynamics from structural motifs;
- Expand on easy-to-replicate tutorials;

Tests
-----

Run nosetests -v to perform tests and diagnoses on functions.


Changelog
---------

v0.1
- Canalization methods ported to Cython

v0.0.4
- Pep8 and python3
- Pinned controllability methods

v0.0.3
- Bugfixes

v0.0.2
- Networkx 2.1 compatibility
- Inclusion of tutorials
- Derrida curve

v.0.0.1
- Control (FVS, MDS, CSTG) methods.
- Canalization methods.
- Implementation ported to public package.