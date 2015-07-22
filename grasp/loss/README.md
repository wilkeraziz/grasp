# Loss

Loss functions can be used in decision rules based on risk as well as in optimisation by empirical risk minimisation.

## BLEU

BLEU is a similarity function, thus in practice we use 1 - BLEU.
This is a sentence-level approximation to BLEU, where approximation is done rather naively by simple smoothing of counts.


# Missing

Parsing losses such as

 * label precision
 * label recall
 

Reordering losses such as

 * Kendall tau
 

Losses based on other MT metrics and/or string editing operations such as
 
 * BEER
 * METEOR
 * TER
 * Levenshtein
 