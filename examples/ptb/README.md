# Important parameters


The example PTB grammar was trained with [discodop](https://github.com/andreasvc/disco-dop).
The following are important parameters when sampling from such a grammar:


```bash

# format of the grammar
# including naming conventions, i.e. grammar files are ${grammar}.lex.gz and ${grammar}.rules.gz
--grammarfmt discodop
# tell Grasp to apply a log transform to the parameters of the PCFG
--log
# the grammar's start symbol
--start TOP
# the tagging model for unknown words
--unkmodel stfd6

```


# Examples


## Exact parser


```bash

# exact parser
time echo -e 'I was given a million dollars .' | python -m grasp.cfg.parser wsj00 dollars --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --viterbi --kbest 100 -v

```

## Randomised parser

    
```bash

# sliced parser with constant shape parameters 
time echo -e 'I was given a million dollars .' | python -m grasp.cfg.parser wsj00 dollars --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --framework slice --prior-a const 0.2 -v

# sliced parser with sampled shape parameters
time echo -e 'I was given a million dollars .' | python -m grasp.cfg.parser wsj00 dollars --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --framework slice --prior-a sym 0.5 -v

```

* some not so interesting settings

        time echo -e 'I was given a million dollars .' | python -m grasp.cfg.parser wsj00 dollars --grammarfmt discodop --start TOP --unkmodel stfd6 --log --samples 200 --viterbi --kbest 100 --framework slice --prior-a beta 2,5 -v
