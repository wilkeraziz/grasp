<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Monolingual parsing](#monolingual-parsing)
  - [Grammar](#grammar)
    - [Start symbol](#start-symbol)
    - [Parameters](#parameters)
    - [File format](#file-format)
    - [Additional grammars](#additional-grammars)
  - [Parser](#parser)
  - [POS tagger](#pos-tagger)
  - [Info](#info)
  - [Viterbi](#viterbi)
  - [Sampling](#sampling)
    - [Slice sampling](#slice-sampling)
  - [Output files](#output-files)
  - [Multiprocessing](#multiprocessing)
- [Profiling](#profiling)
- [Missing](#missing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Monolingual parsing


`grasp.cfg.parser` is the main program, you can use `--help` to get a complete (and long) list of options.
This is an example run:

    
        echo 'I was given a million dollars !!!' | python -m grasp.cfg.parser --log --grammarfmt discodop tests/ptb/wsj00 --unkmodel stfd6 --start TOP --count --samples 1000 tests/ptb/output -v


For more details about the options, check the instructions below.

## Grammar

### Start symbol

By defaul we expect grammars to have a single start symbol (S), 
in case you want to specify a different symbol, use `--start LABEL`.

### Parameters

We expect grammar files to contain log probabilities, 
in case you have a file with standard probabilities, 
use the switch `--log` to tell the parser to apply the log transform.

### File format

We support 2 grammar formats, one inspired by Moses and cdec (which we call 'bar'), 
and the format produced by [discodop](https://github.com/andreasvc/disco-dop), 
you can switch between formats using `--grammarfmt FMT`. 
The default format is 'bar', in which case the grammar is expected to be in a single file. 
Discodop splits the grammar into two files, namely, a set of rules and the lexicon. 
If you are using discodop's grammars, then all you need to provide is the prefix to the grammar, 
the parser will complete it with '.rules.gz' and '.lex.gz'.


### Additional grammars

`Grasp` supports multiple grammars as well as glue grammars.
 Glue rules are special in that they only apply to initial states.
            
            time echo 'I was given a million dollars .' | python -m grasp.cfg.parser tests/cfg/grammar --extra-grammar tests/cfg/hook --glue-grammar tests/cfg/glue --unkmodel passthrough --log --viterbi -v tests/cfg/output
            

* The example above is a simple binary bracketing grammar. 

* The grammars are all in `bar` format and we apply the `log` transform to all of them 
(for now you cannot set these options separately for each grammar).

* The main grammar is `cfg/grammar`, if you use one of the report options below, they will only apply to the main grammar.

* We have an additional grammar `cfg/hook` which from the point of view of the parser is no different from the main grammar.

* Finally, we have a glue grammar, which is special in the sense that its rules can only span from an initial state of the input automaton.

* Both `--extra-grammar` and `--glue-grammar` can be used multiple times to specify a set of grammars and glue grammars.

## Parser

In this toolkit, parsing is done using more general algorithms for intersection between wCFGs and wFSA:
You will find implementations of 2 such algorithms:

* `--intersection nederhof` is CKY+ as in (Nederhof and Satta, 2003)
* `--intersection earley` is the top-down Earley-inspired algorithm in (Dyer and Resnik, 2010)

The advantage of using such algorithms is that support to wFSA input is straightforward.

You might want to choose the label of the goal node after intersection `--goal LABEL`


## POS tagger

You might want the parser to deal with unknown words in special ways. 
We have implemented the switch `--unkmodel MODEL`, where model takes one of the following values:

* a `passthrough` model, here we simply add a dummy rule which produces the unknown word with probability 1.0, 
to configure the LHS of this rule use `--default-symbol LABEL`;
* `stfdbase`, `stfd4` and `stfd6` are models used by the Stanford parser, 
our implementation is a verbatim copy from discodop's;

## Info

You might be interested in some specific information about the parser, the grammar, the chart, etc. 
Here you find some options.

* `--forest` dumps the forest as a grammar (in 'bar' format)
* `--report-top` tries to top-sort the grammar in order to report its start symbols and exit
* `--count` reports the number of derivations in the forest (note that this requires running the inside algorithm)
* `--verbose` increases the verbosity level of the parser
* `--progress` adds a progressbar to lengthy runs

## Viterbi

This parser can enumerate derivations sorted by weight.

* `--viterbi` finds the best derivation using the max-times semiring
* `--kbest K` finds the K best derivations using a lazy enumeration algorithm (Huang and Chiang, 2005); 
use K=1 to retrieve the best derivation using this algorithm instead of the max-times semiring.

## Sampling

You might want to sample from the distribution defined by the forest.

* `--samples N` draws N random samples from the forest

If the framework option is set to *exact*, i.e. `--framework exact`, then this will be an instance of
**ancestral sampling**.
Otherwise, see below.

### Slice sampling

For large grammars, slice sampling might be a better idea `--sampler slice`.

* `--burn N` burns the first N samples
* `--batch K` samples K derivations per iteration (defaults to 1)
* `--resample N` resamples N derivations from each batch (disabled by default)
* `--free-dist TYPE` used to specify the type of distribution associated with free slice variables.

    You can choose a **Beta** distribution, convenient for PCFGs or an **Exponential** distribution, 
    convenient for log-linear models. 
    Their parameters can be made constant or sampled from **Beta** or **Gamma** priors, respectively, see below.
    
* `--prior-a TYPE PARAMETERS`, `--prior-b TYPE PARAMETERS` and `--prior-scale TYPE PARAMETERS` are used to specify a prior distribution (and their parameters) associated with the parameters of the free distribution. 

    Priors can be constant (TYPE=const), a **Beta** (TYPE=beta), or a symmetric **Gamma** (TYPE=gamma).
    If `const`, you must specify a value (e.g. PARAMETERS=0.1).
    If `beta`, you must specify two shape parameters separated by a comma (e.g. PARAMETERS=0.1,1).
    If `gamma`, you mus specify a scale parameter (the shape is assumed 1).

* `--save-chain` is used to save the entire Markov chain to disk


Example:


        time head -n1 tests/ptb/input | python -m grasp.cfg.parser tests/ptb/wsj00 --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --framework slice --prior-a const 0.2  --count --save-chain -v --progress tests/ptb/dollars 

## Output files

The output files are in the workspace. You can name your experiment using `--experiment name`, 
otherwise it will be named by a timestamp and a random suffix.
 
You will find several directories within an experiment tree.

* ancestral 
* viterbi
* kbest
* slice

They store results for each parsing strategy.
Folders for strategies based on sampling contain:

* derivations
* trees
* yields
* chain

Where you will find the corresponding output level of the sampler.

Depending on whether the user requires some reports to be produced, the experiment folder might also contain:

* count
* forest


Output files associated with a certain input segment are named numerically with a 0-based identifier which represents
the order in which the input objects were presented.


## Multiprocessing

You can use `--cpus N` to spawn a number of processes and finish a task quickly.
When using multiple cpus the logging information on screen becomes a mess. I would avoid setting verbosity levels 
with `-v` and `--progress`.


# Profiling

First get a `pstats` report using `--profile report.pstats`. This is a `pstats` object obtained with `cProfile`.
Then analyse the result either on `ipython` or by converting the report to a graph using `gprof2dot`.

    
        gprof2dot -f pstats report.pstats | dot -Tpng -o report.png

        
        
# Missing 

* a simple pruning strategy for Viterbi
* load grammars only once when using multiprocessing

    