.. currentmodule:: cana.boolean_network

Control
=======

These are methods and modules used to calculate control in Boolean Networks. They are divided in dynamics- and structure-based methods.

Note that these methods do not need to be called directly, as :class:`cana.boolean_network.BooleanNetwork` provides the appropriate methods.

.. contents:: Contents
	:depth: 3

Dynamics based control
------------------------

The control methods used here are implemented directly on the base class :class:`.BooleanNetwork` and :class:`.BooleanNode`. That is because the Network class can ask its nodes directly to step into a specific trajectory, thus compartmentalizing the logic.

Attractor Control	
^^^^^^^^^^^^^^^^^

.. automethod:: cana.boolean_network.BooleanNetwork.attractor_driver_nodes
	:noindex:

.. automethod:: cana.boolean_network.BooleanNetwork.controlled_state_transition_graph
	:noindex:

.. automethod:: cana.boolean_network.BooleanNetwork.controlled_attractor_graph
	:noindex:

Structure based control
-----------------------

These are control methods that only take the structure of the boolean network (aka: the structure graph) into consideration when computing driver nodes.

.. automodule:: cana.control.fvs
	:members:
	
.. automodule:: cana.control.mds
	:members:

.. automodule:: cana.control.sc
	:members: