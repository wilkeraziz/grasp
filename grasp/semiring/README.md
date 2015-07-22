# Real-valued semirings

These are semirings with numerical intepretation such as

 * Boolean
 * Counting
 * Viterbi (which we prefer calling MaxTimes to avoid confusion with the Viterbi algorithm)
 * Inside (which we prefer calling SumTimes to avoid confusion with the Inside algorithm)
 

# Other semirings

Several of the more abstract semirings correspond to a classic inference algorithm:
 * 1-best semiring: Viterbi derivation
 * k-best semiring: k-best derivations
 * sample semiring: ancestral sampling
 * forest semiring: weighted intersection
 
Some of these semirings are defined over pairs, such as
 * a derivation and its value in the max-times semiring (useful in finding the Viterbi derivation)
 * a derivation and its value in the sum-times semiring (useful in sampling derivations)
 
Instead of implementing these semirings directly, due to efficiency reasons, 
`Grasp` implements the corresponding inference algorithms and it consults a precomputed forest 
for item derivations and their values.

Check the package `grasp.inference` for more.
 
