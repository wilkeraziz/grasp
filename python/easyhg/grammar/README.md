# Monolingual parsing


`parse.py` is the main program, you can use `--help` to get a complete list of options.
This is an example run:

    
        echo 'I was given a million dollars !!!' | python parse.py --log --grammarfmt discodop ../../tests/ptb/wsj00 --unkmodel stfd6 --start TOP --count -v --samples 1000


For more details about the options, check the instructions below.

## Grammar

* By defaul we expect grammars to have a single start symbol (S), in case you want to specify a different symbol, use `--start LABEL`.

* We expect grammar files to contain log probabilities, in case you have a file with standard probabilities use the switch `--log` to tell the parser to apply the log transform.

* We support 2 grammar formats, one inspired by Moses and cdec (which we call 'bar'), and the format produced by [discodop](https://github.com/andreasvc/disco-dop), you can switch between formats using `--grammarfmt FMT`. The default format is 'bar', in which case the grammar is expected to be in a single file. Discodop splits the grammar into two files, namely, a set of rules and the lexicon. If you are using discodop's grammars, then all you need to provide is the prefix to the grammar, the parser will complete it with '.rules.gz' and '.lex.gz'.

## Parser

In this toolkit, parsing is done using more general algorithms for intersection between wCFGs and wFSA:
You will find implementations of 3 such algorithms:

* `--intersection nederhof` is CKY+ as in (Nederhof and Satta, 2003)
* `--intersection cky` is CKY+ as in (Dyer, 2010)
* `--intersection earley` is the top-down Earley-inspired algorithm in (Dyer and Resnik, 2010)

The advantage of using such algorithms is that support to wFSA input is straightforward.

You might want to choose the label of the goal node after intersection `--goal LABEL`

## POS tagger

You might want the parser to deal with unknown words in special ways. We have implemented the switch `--unkmodel MODEL`, where model takes one of the following values:

* a `passthrough` model, here we simply add a dummy rule which produces the unknown word with probability 1.0, to configure the LHS of this rule use `--default-symbol LABEL`;
* `stfdbase`, `stfd4` and `stfd6` are models used by the Stanford parser, our implementation is a verbatim copy from discodop's;

## Info

You might be interested in some specific information about the parser, the grammar, the chart, etc. Here you find some options.

* `--forest` dumps the forest as a grammar (in 'bar' format)
* `--report-top` tries to top-sort the grammar in order to report its start symbols and exit
* `--count` reports the number of derivations in the forest (note that this requires running the inside algorithm)
* `--verbose` increases the verbosity level of the parser

## Viterbi

This parser can enumerate derivations sorted by weight.

* `--kbest K` specifies the number of derivations, use K=1 to retrieve the Viterbi derivation

## Sampling

You might want to sample from the distribution defined by the forest.

* `--samples N` draws N random samples from the forest
* `--sampler ALG` specifies a sampling algorithm

Use `--sampler ancestral` (which is also the default) to sample exactly from the forest.

### Slice sampling

For large grammars, slice sampling (Blunsom and Cohn, 2010) might be a better idea `--sampler slice`.

* `--burn N` burns the first N samples
* `--batch K` samples K derivations per iteration (defaults to 1)
* `--beta-a FIRST OTHERS` specify the first parameter of the Beta distribution BEFORE and AFTER the first derivation is found
* `--beta-b FIRST OTHERS` specify the second parameter of the Beta distribution BEFORE and AFTER the first derivation is found
* `--heuristic STRATEGY` specifies a heuristic to find an initial derivation
    * `empdist` samples slice variables from the conditional p(rhs|lhs)
    * `uniform` samples slice variables uniformly within an interval (see below) 
* `--heuristic-empdist-alpha FLOAT` peaks the distribution for the heuristic `empdist`
* `--heuristic-uniform-params LOWER UPPER` the interval for the heuristic `uniform` specified in terms of percentiles of the condition p(rhs|lhs)

The intuition behind the heuristics is to heavily prune the number of rules for each LHS symbol. 
These heuristics turned out to be quite innefective either prunning to heavily to the point that no parse can be found or prunning too little to the point that the first iteration is too slow (and might still result in a forest without any parses).

For now slice sampling only supports `--intersection nederhof`.


## Ongoing work

* better heuristics for the initial derivation in slice sampling
* decision rules such as `--map` and `--mbr METRIC`
* loss-augmented and loss-diminished decoding
* a simple pruning strategy for Viterbi
* Gibbs sampler (Bouchard-Cote et al, 2009).
* Importance sampler (Aziz, 2015)


# Profiling

First get a `pstats` report using `cProfile`


        echo 'I was given a million dollars .' | python -m cProfile -o NEDERHOF.pstats parse.py --log --grammarfmt discodop ../../tests/ptb/wsj00 --unkmodel stfd6 --count -v --kbest 1 --intersection nederhof


Then convert the report to a graph using `gprof2dot`

    
        gprof2dot -f pstats NEDERHOF.pstats | dot -Tpng -o NEDERHOF.png


