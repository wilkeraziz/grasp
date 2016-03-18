# Inference

Several inference algorithms here have corresponding abstract semirings.
They mostly operate over derivations or sets of derivations whose values are represented in a real-valued semiring.

# Value

Implements the value recursion in a given semiring consulting item derivations in a precomputed forest.
Our implementation is robust to cycles, it uses an iterative approximation to the supremum (see Goodman, 1999).

# Algorithms

Having compute node values and edge values (using the value recursion), one can obtain derivations as follows. 

### Viterbi

This returns the best-derivation, it assumes values are defined in the max-times semiring.
 
## Sample

This samples a random derivation, it assumes values are defined in the sum-times semiring.
We provide a wrapper called `AncestralSampling`, which takes care of value computations and 
provides a few extra functionalities.

## K-best

This is enumerates k derivations in best-first order. We have an implementation of Huang and Chiang (2005).