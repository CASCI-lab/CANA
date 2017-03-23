
Notation
==========


This notation follows the initial notation in Marques-Pita & Rocha [2013] from where all subsequent paper derive.


Node (automata)
---------------

.. |x| replace:: :math:`x`
.. |k| replace:: :math:`k`
.. |F| replace:: :math:`F`
.. |f_a| replace:: :math:`f_\alpha`
.. |c_a| replace:: :math:`c_{\alpha}`
.. |s_a| replace:: :math:`s_{\alpha}`

+--------+-------------------------------------------------------------------------------+
| symbol | description                                                                   |
+--------+-------------------------------------------------------------------------------+
| |x|    | Boolean Automaton (:class:`.BooleanNode`)                                     |
|        |                                                                               |
|        | Binary-state automaton                                                        |
+--------+-------------------------------------------------------------------------------+
| |k|    | Number of inputs of |x|                                                       |
|        |                                                                               |
|        | How many inputs determine the transitions of an automaton |x|                 |
+--------+-------------------------------------------------------------------------------+
| |F|    | Look-up table (LUT) of |x|                                                    |
|        |                                                                               |
|        | The transition function of |x| represented as a LUT (:math:`2^k` entries)     |
+--------+-------------------------------------------------------------------------------+
| |f_a|	 | LUT entry in |F|                                                              |
|        |                                                                               |
|        | k-tuple combination of input states                                           |
|        | i.e. condition and corresponding transition                                   |
+--------+-------------------------------------------------------------------------------+
| |c_a|  | condition part in a LUT entry of |f_a|                                        |
|        |                                                                               |
|        | k-tuple combination of input states i.e. condition                            |
+--------+-------------------------------------------------------------------------------+
| |s_a|	 | transition in a LUT entry |f_a|                                               |                      
|        |                                                                               |
|        |Boolean state prescribed as the transition in |f_a|                            |
+--------+-------------------------------------------------------------------------------+

Network
----------

.. |B| replace:: :math:`B`
.. |X| replace:: :math:`X`
.. |n| replace:: :math:`n`
.. |*x*| replace:: :math:`\textbf{x}`
.. |X_i| replace:: :math:`X_i`
.. |k_i| replace:: :math:`k_i`
.. |F_i| replace:: :math:`F_i`
.. |f_ia| replace:: :math:`f_{i:\alpha}`
.. |A_i| replace:: :math:`A_i`
.. |o(x)| replace:: :math:`\sigma(\textbf{x}) \rightarrow A`

.. |a| replace:: :math:`\alpha`
.. |A| replace:: :math:`A`
.. |i| replace:: :math:`i`

+--------+-------------------------------------------------------------------------------+
| symbol | description                                                                   |
+--------+-------------------------------------------------------------------------------+
| |B|    | Boolean Network (:class:`.BooleanNetwork`)                                    |
|        |                                                                               |
|        | A graph of |N| automata with directed edges (source node is input of end node)|
+--------+-------------------------------------------------------------------------------+
| |X|    | set of automata in |B|                                                        |
|        |                                                                               |
|        | set of Boolean automata that constitute a Boolean Network |B|                 |
+--------+-------------------------------------------------------------------------------+
| |n|    | number of nodes in |B|                                                        |
|        |                                                                               |
|        | :math:`n = |X|`                                                               |
+--------+-------------------------------------------------------------------------------+
| |*x*|  | network configuration                                                         |
|        |                                                                               |
|        | collection of the states of all nodes in a Boolean Networ :math:`B`           |
+--------+-------------------------------------------------------------------------------+
| |X_i|  | set of inputs of nodes |x_i| in |B|                                           |
|        |                                                                               |
|        | Set of input nodes of |x_i|                                                   |
+--------+-------------------------------------------------------------------------------+
| |k_i|  | in-degree of |x_i|                                                            |
|        |                                                                               |
|        | Cardinality of |X_i|                                                          |
+--------+-------------------------------------------------------------------------------+ 
| |F_i|  | Look-up table of |x_i|                                                        |
|        |                                                                               |
|        | Transition function represented as a LUT.                                     |
+--------+-------------------------------------------------------------------------------+
| |f_ia| | LUT entry in |F_i|                                                            |
|        |                                                                               |
|        | Sub-indices |i| and |a|, separated by ’:’, are used to specify node and entry.|
+--------+-------------------------------------------------------------------------------+
| |A_i|  | An attractor of |B|                                                           |
|        |                                                                               |
|        | A specific (index |i|) fixes-point or periodic attractor of a B.N. |B|        |
+--------+-------------------------------------------------------------------------------+
| |o(x)| | Dynamic trajectory of |*x*| to |A|                                            |
|        |                                                                               |
|        | This notation is used to represent that the trajectory of some                |
|        | configuration |*x*| is known to converge to |A|                               |
+--------+-------------------------------------------------------------------------------+


Wildcard
----------

.. |#| replace:: :math:`\#`
.. |F'| replace:: :math:`F^{'}`
.. |f'v| replace:: :math:`f^{'}_{v}`
.. |Yv| replace:: :math:`\Upsilon_{v}`

.. |faeF| replace:: :math:`f_{\alpha} \in F`

+--------+-------------------------------------------------------------------------------+
| symbol | description                                                                   |
+--------+-------------------------------------------------------------------------------+
| |#|    | Wildcard symbol                                                               |
|        |                                                                               |
|        | If this symbol appears in a condition, the variable it represents can         |
|        | be in any state                                                               |
+--------+-------------------------------------------------------------------------------+
| |F'|   | wildcard-schema redescription of |F|                                          |
|        |                                                                               |
|        | LUT where entries are wildcard schemata                                       |
+--------+-------------------------------------------------------------------------------+
| |f'v|  | a wildcard schema in |F'|                                                     |
|        |                                                                               |
|        | An entry in |F'| is like an entry in |F| but its condition part can           |
|        | have wildcard symbols                                                         |
+--------+-------------------------------------------------------------------------------+
| |Yv|   | Entries |faeF| in |f'v|                                                       |
|        |                                                                               |
|        | collection of the states of all nodes in a Boolean Network |B|                |
+--------+-------------------------------------------------------------------------------+


Two-Symbol
-----------

# TODO

