# Monolingual parsing


`parse.py` is the main program, you can use `--help` to get a complete list of options.

This gives an idea of the basic command:


      echo '1 2 3 4' | python parse.py ../../example/binary --pass-through


There are 3 algorithms for intersection between wCFGs and wFSAs.
In this toolkit, such algorithms are used for monolingual parsing:

* `--intersection nederhof` is CKY+ as in (Nederhof and Satta, 2003)
* `--intersection cky` is CKY+ as in (Dyer, 2010)
* `--intersection earley` is the top-down Earley-inspired algorithm in (Dyer and Resnik, 2010)

The advantage of using such algorithms is that support to wFSA input is straightforward.

You can use different inference algorithms:

* Sampling `--samples 1000`
* Viterbi `--kbest 1`
* K-best `--kbest 100`

I am soon going to implement a few decision rules on top of sampling (e.g. `--map` and `--mbr {metric}`).

I might also add support to loss-augmented and loss-diminished decoding.

I might add support to some simple pruning strategy.

You can also use grammars trained by [discodop](https://github.com/andreasvc/disco-dop)

    
        echo 'I was given a million dollars !!!' | python parse.py --log --grammarfmt discodop /Users/waziz/workspace/ptb/WSJ/ORIGINAL_READABLE_CLEANED/bintrees/sample/pcfg --unkmodel stfd6 --count --intersection nederhof -v --samples 1000



## MCMC monolingual parsing

`mcmcparse.py` implements an MCMC approach to parsing. The difference between this and `parse.py` is that here a complete chart is (probably) never instantiated.

The basic command looks like the following:

    echo '1 2 3 4' | python mcmcparse.py ../../example/binary --pass-through --beta-a 0.5 --samples 1000


For now it only supports `--intersection nederhof`, but `--intersection earley` is on its way.

For now it only implements the slice sampler proposed by Blunsom and Cohn (2010).
I do intend to add 2 more methods, namely, the Gibbs sampler in (Bouchard-Cote et al, 2009) and a new importance sampler.

For exact sampling from the forest check `parse.py` with the option `--samples {n}`.


        time echo "these parts are provided to conduct rainwater" | python mcmcparse.py /Users/waziz/workspace/github/pcfg-sampling/examples/reordering/en-ja.pcfg --intersection nederhof --pass-through --log --default-symbol UNK --start ROOT -v --samples 10000 --beta-a 0.1 1.4 --beta-b 0.4 1.2 --batch 100 > x
