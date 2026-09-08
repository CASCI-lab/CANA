CANAlization: Control & Redundancy in Boolean Networks
=======================================================

This package implements a series of methods used to study control, canalization and redundancy in Boolean Networks.

Citation:
-------------

If you use `cana` in your research, please cite us and check out our related papers below! 

- A.M. Marcus, J.C. Rozum, H. Sizek, L.M. Rocha [2025]. "[CANA v1.0.0: efficient quantification of canalization in automata networks](https://doi.org/10.1093/bioinformatics/btaf461)". *Bioinformatics*. btaf461. doi: 10.1093/bioinformatics/btaf461

- R.B. Correia, A.J. Gates, X. Wang, L.M. Rocha [2018]. "[CANA: A python package for quantifying control and canalization in Boolean Networks](https://www.informatics.indiana.edu/rocha/publications/FSB18.php)". *Frontiers in Physiology*. **9**: 1046. doi: 10.3389/fphys.2018.01046

Installation:
-------------

**Latest stable release**
```
    pip install cana
```

**Manuscript-specific version**
```
    pip install cana=1.0.0
```

**Latest development release on GitHub**
```
    pip install git+https://github.com/CASCI-lab/CANA
```

Please note that CANA uses Cython. For it to compile you may need to install the following:
```pip install Cython```

Docs:
-------

The full documentation can be found at: [casci-lab.github.io/CANA/](https://casci-lab.github.io/CANA/)


Papers with the Theory, Formulations, and Analytical Examples:
---------

- F.X. Costa, J.C. Rozum, A.M. Marcus, L.M. Rocha [2023]. "[Effective Connectivity and Bias Entropy Improve Prediction of Dynamical Regime in Automata Networks](https://doi.org/10.3390/e25020374)". *Entropy*. 25(**2**):374. doi: 10.3390/e25020374.

- S. Manicka, M. Marques-Pita, L.M. Rocha [2022]. "[Effective connectivity determines the critical dynamics of biochemical networks](https://doi.org/10.1098/rsif.2021.0659)". *Journal of the Royal Society Interface*. 19(**186**)20210659. doi: 10.1098/rsif.2021.0659.

- A.J. Gates, R.B. Correia, X. Wang, L.M. Rocha [2021]. "[The effective graph reveals redundancy, canalization, and control pathways in biochemical regulation and signaling](https://doi.org/10.1073/pnas.2022598118)". *Proceedings of the National Academy of Sciences (PNAS)*. 118(**12**). doi: 10.1073/pnas.20225981186

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

- Kyu Hyong Park, github: [kyuhyongpark](https://github.com/kyuhyongpark)
- Yoshiaki Fujita, github: [yoshiakifujita](https://github.com/yoshiakifujita)
- Jordan C. Rozum, github: [jcrozum](https://github.com/jcrozum)
- Felipe Xavier Costa, github: [fxcosta-phd](https://github.com/fxcosta-phd)
- Austin Marcus, github: [austin-marcus](https://github.com/austin-marcus)
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


Changelog
---------

Master
- Added visualization routines to drawing (effective graph and conditional effective graph)

v1.0.1
- Add Python 3.13 support (schematodes 1.0.1, PyO3 0.22)

v1.0.0
- A.M. Marcus, J.C. Rozum, H. Sizek, L.M. Rocha [2025]. "[CANA v1.0.0: efficient quantification of canalization in automata networks](https://doi.org/10.1093/bioinformatics/btaf461)". *Bioinformatics*. btaf461. doi: 10.1093/bioinformatics/btaf461

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
