<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Hierarchical MT decoding](#hierarchical-mt-decoding)
  - [Input](#input)
  - [Grammar](#grammar)
  - [Model](#model)
  - [Exact decoding](#exact-decoding)
  - [Slice sampling](#slice-sampling)
- [Decision rule](#decision-rule)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Hierarchical MT decoding

`grasp.mt.decoder` is the main program, you can use `--help` to get a complete (and long) list of options.

This is an example run:


        time head -n2 tests/mt/input | python -m grasp.mt.decoder --grammars tests/mt/grammars --glue-grammar tests/mt/glue --pass-through --rt --slm StatelessLanguageModel 2 tests/mt/btec.klm2 --weights tests/mt/mira/lm2-q --temperature 0.1 --count --forest --viterbi --kbest 100 --samples 1000 tests/mt/output


## Input

We use `cdec` style segments, that is, they are annotated with information such as path to grammar.

        <seg grammar="grammars/grammar.0.gz" id="0">一 跳 一 跳 的 痛 。</seg>


## Grammar

The grammar is a SCFG extracted by `cdec`. You need to do it beforehand.
Additionally we also have a glue grammar `--glue-grammar tests/mt/glue`.

Some words are unknown and the parser will not know what to do with them.
Here the only option you have is to add fake rules which pass the unknown word through the translation `--pass-through`.

Because we are not running this command from the directory `tests/mt`, we need to redirect the decoder to look for grammars
in a different location: `--grammars tests/mt/grammars`.

## Model

This is a decoder for a log-linear model, you can choose from a few features or contribute your own.
For more specific information about feature extraction check `grasp.ff`, here you will only find how to instantiate
some of them through the command line interface.

* `--rt` is a rule table feature extractor for the default cdec features, it basically looks for key-value pairs in the grammar file
* `--slm StatelessLanguageModel 2 tests/mt/btec.klm2` this is a stateless LM feature, which simply scores ngrams within rules taken in isolation

These extractors is all you need to reproduce a system like `cdec` (just `--rt`) or `Moses` (both `--rt` and `--slm`) without a language mode.

The following are features which are typically used in a rescoring pass.

* `--wp WordPenalty -0.43429466666666666666` this is a penalty associated with each target word, you need to provide a name for this extractor and the value of the penalty (here we used the value used in hiero)
* `--ap Arity 1` this is a binary feature which indicates the arity of a rule; in older versions `cdec` used to deal with
 it as a penalty-like feature, in which case you would use `--ap -0.43429466666666666666` (as in hiero)
* `--lm LanguageModel 2 tests/mt/btec.klm2` this is a bigram language model.
    You need to give it a name (e.g. LanguageModel), the order of the model (e.g. 2), and a path to kenlm model.

Note that the word penalty and the arity penalty are not stateful features, however, they are typically dealt with
 in the rescoring phase because of their correlation with the language model.


The last two things to specify a model are model weights and a temperature (relevant in sampling).

* `--weights mira/lm2-p` this is a file with feature names and feature weights (that's why you had to name the extractors)
* `--temperature 0.1` peaks or flattens the model (because if you learnt your model using MIRA the scaling was arbitrary).


More examples

* no language model

            time head -n2 tests/mt/input | python -m grasp.mt.decoder --grammars tests/mt/grammars --glue-grammar tests/mt/glue --pass-through --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --slm StatelessLanguageModel 2 tests/mt/btec.klm2 --weights tests/mt/mira/lm2-q --temperature 0.1 --count --forest --viterbi --kbest 100 --samples 1000 tests/mt/output

* language model (with exhaustive intersection)

            time head -n2 tests/mt/input | python -m grasp.mt.decoder --grammars tests/mt/grammars --glue-grammar tests/mt/glue --pass-through --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --lm LanguageModel 2 tests/mt/btec.klm2 --weights tests/mt/mira/lm2-p --temperature 0.1 --count --forest --viterbi --kbest 100 --samples 1000 tests/mt/output

Do note that these examples use different weight files (because they have different components).


 --count --forest --viterbi --kbest 100 --samples 1000 tests/mt/output


## Exact decoding

You will only be able to use without stateful components (e.g. language model) or for very toy experiments (short sentences and at most a bigram LM).
The relevant option is `--framework exact`, then you have access to algorithms such as

* Viterbi `--viterbi`
* K-best `--kbest K`
* Ancestral sampling `--samples N`

## Slice sampling

You can use slice sampling whenever you have a stateful scorer such as the language model.

* `--lag N` collected evenly spaced samples
* `--burn N` discard the initial samples
* `--batch B` how many samples are used to estimate the sliced distribution (see below)
* `--chains N` heuristically start N chains with N random seeds and average them in the end (note that this has no guarantee of increased variance)
* `--within alg` which algorithm to use in sampling from the slice (see below)
* `--save-chain` stores the entire Markov chain
* `--temperature0 alpha` at which temperature should we collect random seeds (you typically want to increase this number, so that seeds are pretty random)
* `--prior TYPE PARAMETERS` this is the type of prior on the scale of the **Exponential** distribution which controls free slice variables.
    Examples are 'const 1' which is a constant scale set to 1, `sym 1` to sample the scale from a symmetric Gamma with scale 1,
    or `asym 99` to estimate an asymmetric Gamma whose scale parameters are based on the last percentile of the distribution
    of incoming edges associated with each slice variable (this does not seem to work very well, the other two options are much better).

First, check the paper in order to understand the method.
   Then, you will understand that we can either sample *exactly* from the slice or *approximately*.
   In the first case, we rescore the entire slice and sample by **ancestral sampling**.
   This can be very expensive, particularly, for higher-order LMs.
   In the second case, we can resample the slice uniformly and rescore the sample, that would be `--within uniform`.
   This has proven itself a good solution.
   Another solution, which seems to work even better, is to importance sample the slice, that would be `--within importance`.
   When sampling approximately, you will need to specify `--batch` which somehow controls the quality of the approximation.

Example:


        time head -n2 tests/mt/input | python -m grasp.mt.decoder --grammars tests/mt/grammars --glue-grammar tests/mt/glue --pass-through --rt --wp WordPenalty -0.43429466666666666666 --ap Arity -0.43429466666666666666 --lm LanguageModel 2 tests/mt/btec.klm2 --weights tests/mt/mira/lm2-p --temperature 0.1 --samples 1000 --framework slice --batch 100 --within uniform --temperature0 10 --prior const 1 tests/mt/output --experiment uniform --progress


# Decision rules

A few decision rules can be applied after sampling regardless of the strategy you choose.


        python -m grasp.mt.decision tests/mt/output/uniform/slice/yields --rule consensus tests/mt/output/uniform/slice/yields/consensus


You need to specify where sampled translations (*yield* not *derivation*) are read from (e.g. ` tests/mt/output/uniform/slice/yields`)
and where decisions are written to (e.g. `tests/mt/output/uniform/slice/yields/consensus`).
Also you can choose between `--rule map` (the most frequent string in the sample), `--rule mbr` (the MBR solution) and `--rule consensus` (an approximation to the MBR solution which runs in linear time).
The risk-based rules (MBR and consensus) use a loss based on the given metric `--metric M` (which defaults to BLEU with plus 1 smoothing).

This will produce

* `tests/mt/output/uniform/slice/consensus.best` which is the file containing the selected translations
* `tests/mt/output/uniform/slice/consensus.complete.gz` which contains the entire ranking based on the samples
