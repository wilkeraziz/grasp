# Hypergraph

A backward hypergraph is an excellent data structure to compactly and efficiently organise a forest.
In `Grasp`, a CFG is a more pythonic object whose underlying container is a dictionary.
A Hypergraph, on the other hand, has a more C-like interface.
 Its underlying containers are a mixture of python and C data structres and it is designed for direct access.

## Symbol

Any python object, typically immutable and hashable. Examples: Terminal, Nonterminal, Span.

## Rule

Any python object, typicically immutable and hashable. Rules add meaning to the relations expressed in the hypergraph
in the form of edges (see below). Examples: CFGProduction, SCFGProduction.
 
## Node

A node is a container for a Symbol. The guarantees a 1-to-1 correspondence between nodes and symbols.

## Edge

An edge connects a (possibly empty) sequence of nodes, the edge's tail, to a single node, the edge's head, with a weight.
Each edge is associated with a Rule. Unlike nodes, there is no 1-to-1 mapping between edges and rules.
This means that if a rule is added twice to a hypergraph, two equivalent edges will be created 
(other than useless redundancy, this typically has no negative implications).

## Backward-star

The backward-star (BS) of a node is the set of incoming edges to a node. In this framework, the BS is represented
as a list instead of a set (but Hypergraph guarantees no duplicates).

## Interface

Hypergraph does not expose nodes and edges directly, instead it exposes corresponding indices which can be used
to retrieve underlying objects such as Symbol and Rule. 

# DFA

Deterministic finite-state automaton.

# Topological sorting

Two variants:
 * acyclic
 * robust to cycles: based on Tarjan's algorithm for strongly connected components