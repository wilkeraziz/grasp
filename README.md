<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [`Grasp` - Randomised Semiring Parsing](#grasp---randomised-semiring-parsing)
- [Build](#build)
- [Uses](#uses)
- [Name](#name)
- [Citation](#citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# `Grasp` - Randomised Semiring Parsing

A suite of algorithms for inference tasks over (finite and infinite) context-free sets.
For generality and clarity, `Grasp` uses the framework of semiring parsing with support to the most common semirings:
 * Boolean
 * Counting
 * Inside
 * Viterbi
 * 1-best
 * k-best
 * Forest

Parsing is deployed in terms of weighted deduction allowing for arbitrary weighted finite-state input.
`Grasp` contains implementations of both bottom-up (CKY-inspired) and top-down (Earley-inspired) algorithms.

Inference is achieved by Monte Carlo (and Markov chain Monte Carlo) methods such as
 * ancestral sampling
 * slice sampling

In principle, sampling methods can deal with models whose independence assumptions are weaker than what
is feasible by standard dynamic programming.

`Grasp` is meant to help lower the implementation burden in applications such as monolingual constituency parsing,
synchronous parsing, context-free models of reordering for machine translation, and machine translation decoding.

# Build

`Grasp` is written in `python3`. It is recommended to use `virtualenv`.


```bash
virtualenv -p python3 ~/workspace/envs/grasp
source ~/workspace/envs/grasp/bin/activate
```


A few dependencies which `setup.py` does not list, because `setuptools` does not deal very well with them:

 * numpy
 * scipy

Dependencies which `setup.py` will install for you:

 * tabulate
 * ply
 * nltk

Additional dependencies which you will have to install by yourself:

 * [kenlm](https://github.com/kpu/kenlm.git)


If you are contributing to grasp, you can install it in develop mode.


```bash
python setup.py develop
```


Otherwise, just run setup install.

```bash
python setup.py install
```


## Sanity check

This is what my own virtualenv looks like if I list the installed packages with `yolk -l` (to install yolk just run `pip install yolk3k`):

```bash
(grasp)waziz@u036503:~/workspace/github/grasp$ yolk -l
Cython          - 0.22.1       - active
gnureadline     - 6.3.3        - active
grasp           - 0.0.dev1     - active development (/Users/waziz/workspace/github/grasp)
ipython         - 3.2.1        - active
kenlm           - 0.0.0        - active
nltk            - 3.0.4        - active
numpy           - 1.9.2        - active
pip             - 7.1.0        - active
ply             - 3.6          - active
scipy           - 0.15.1       - active
setuptools      - 11.0         - active
tabulate        - 0.7.5        - active
yolk3k          - 0.8.7        - active
```

Note: `Cython` is not a dependency.

# Uses


* [monolingual parsing](grasp/cfg/README.md)
* [hierarchical MT decoding](grasp/mt/README.md)


# Name

I meant to call it `Rasp`, but it turns out the name was taken :p
if you have a good suggestion which justifies the G in `Grasp` please let me know.


# Citation

`Grasp` has not yet been published, but there are two papers on the way.
One describes the toolkit and has been submitted to MTM-PBML 2015.
Another describes the underlying methodology and is to be submitted to TACL by September 2015.

Please do not report experiments using `Grasp` before one of these publications goes through.

`Grasp` is developed by [Wilker Aziz](http://wilkeraziz.github.io) at the University of Amsterdam.
