
Notation
==========


This notation follows the initial notation in :cite:`Marques-Pita:2013` from where all subsequent papers derive.


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
|        | Boolean state prescribed as the transition in |f_a|                           |
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

.. |o_m| replace:: :math:`\circ_m`
.. |b| replace:: :math:`\beta`
.. |F''| replace:: :math:`F''`
.. |f''th| replace:: :math:`f''_{\theta}`
.. |Tht| replace:: :math:`\Theta_{\theta}`
.. |Thtp| replace:: :math:`\Theta'_{\theta}`
.. |Xl| replace:: :math:`X_l`
.. |etal| replace:: :math:`\eta_l`
.. |Xls| replace:: :math:`X_l^s`
.. |Xg| replace:: :math:`X_g`
.. |Xgs| replace:: :math:`X_g^s`
.. |eta| replace:: :math:`\eta`
.. |ng| replace:: :math:`n_g`
.. |ngs| replace:: :math:`n_g^s`

.. |thetao| replace:: :math:`\Theta'_{\theta} = \{ f'_{\alpha} : f_{\alpha} \rightarrow f''_{\theta} \}`
.. |thetinha| replace:: :math:`\Theta'_{\theta} = \{ f'_v : f_v \rightarrow f''_{\theta} \}`

+--------+-------------------------------------------------------------------------------+
| symbol | description                                                                   |
+--------+-------------------------------------------------------------------------------+
| |o_m|  | Position-free symbol                                                          |
|        |                                                                               |
|        | If a variable in the condition part of a schema is marked with this symbol, it|
|        | can exchange places with any other variable in the same schema marked with the|
|        | same symbol. Index :math:`m` is used to distinguish subsets of identically-   |
|        | marked inputs.                                                                |
+--------+-------------------------------------------------------------------------------+
| |b|    | Depth of search for two-symbol schemata                                       |
|        |                                                                               |
|        | Defines the minimum number of wildcard schemata in a two-symbol redescription.|
+--------+-------------------------------------------------------------------------------+
| |F''|  | 2-Symbol redescription of |F|.                                                |
|        |                                                                               |
|        | LUT where entries are two-symbol schemata.                                    |
+--------+-------------------------------------------------------------------------------+
| |f''th|| a 2-Symbol schema in |F''|                                                    |
|        |                                                                               |
|        | An entry in |F''| is like an entry in :math:`|F|` but its condition part can  |
|        | have wildcard and position-free symbols.                                      |
+--------+-------------------------------------------------------------------------------+
| |Tht|  | Entries :math:`f_{\alpha} \in F : f'_v \rightarrow F''_{\theta}`              |
|        |                                                                               |
|        | The set of original LUT entries in :math:`F` redescribed by a single 2-symbol |
|        | schema |thetao|                                                               |
+--------+-------------------------------------------------------------------------------+
| |Thtp| | Schemata :math:`f'_v \in F' : f'_v \rightarrow f''_{\theta}`                  |
|        |                                                                               |
|        | The set of wildcard schemata in F' redescribed by a single 2-symbol schema    |
|        | |thetinha|.                                                                   |
+--------+-------------------------------------------------------------------------------+
| |Xl|   | set of liberals enputs in a schema :math:`f''`                                |
|        |                                                                               |
|        | The variables in the condition part of schema :math:`f''` that are specified  |
|        | in a Boolean state (not wildcard)                                             |
+--------+-------------------------------------------------------------------------------+
| |etal| | size of literal-emput set in schema :math:`f''`.                              |
|        |                                                                               |
|        | :math:`\eta_{l} = |X_{l}|`                                                    |
+--------+-------------------------------------------------------------------------------+
| |Xls|  | :math:`\text{state-}s` literal enputs in :math:`f''`                          |
|        |                                                                               |
|        | Subset :math:`X_l^s \subset X_l` of literal enput in a scpecific state        |
|        | :math:`s : s \in {0,1}`.                                                      |
+--------+-------------------------------------------------------------------------------+
| |Xg|   | group invariant enput in a schema :math:`f''`                                 |
|        |                                                                               |
|        | The set variables in the condition part of schema :math:`f''` that are marked |
|        | with an identical position-free symbol, in every state they can take.         |
+--------+-------------------------------------------------------------------------------+
| |Xgs|  | elements of |Xg| in state :math:`s`                                           |
|        |                                                                               |
|        | This notation is used to refer to the members of a group-invariant enput      |
|        | instantiated in a specific state :math:`s`, that is                           |
|        | :math:`X_g^s = { \forall x_i \in X_g \land x_i = s}`.                         |
+--------+-------------------------------------------------------------------------------+
| |eta|  | number of group-invariant enputs in :math:`f''`                               |
|        |                                                                               |
|        | Number of subsets of inputs marked with a distinct position-free symbol.      |
+--------+-------------------------------------------------------------------------------+
| |ng|   | size of a single group-invariant enput :math:`g` in :math:`f''`               |
|        |                                                                               |
|        | Number of inputs marked with the position-free symbol in :math:`g`.           |
+--------+-------------------------------------------------------------------------------+
| |ngs|  | a sub-constraint in :math:`X_g` on state :math:`s \in {0,1}`                  |
|        |                                                                               |
|        | specifies a group-invariant constraint in the set :math:`X_g`, at least       |
|        | :math:`n_g^s` variables must be in state :math:`s`.                           |
+--------+-------------------------------------------------------------------------------+

