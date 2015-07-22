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

`Grasp` is meant to help lower the implementation burned in applications such as monolingual constituency parsing, 
synchronous parsing, context-free models of reordering for machine translation, and machine translation decoding.

# Build

`Grasp` is written in `python3`. It is recommended to use `virtualenv`.

        
        virtualenv -p python3 ~/workspace/envs/grasp
        source ~/workspace/envs/grasp/bin/activate


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

        
        python setup.py develop


Otherwise, just run setup install. 


        python setup.py install


# Name

I mean to call it `Rasp`, but it turns out the name was taken :p
if you have a good suggestion which justifies the G in `Grasp` please let me know.


# Citation

`Grasp` has not yet been published yet, but there are two papers on the way. 
One describes the toolkit and has been submitted to MTM-PBML 2015.
Another describes the underlying methodology and is to be submitted to TACL by September 2015.

Please do not report experiments using Graps before one of these publications goes through.

`Grasp` is developed by [Wilker Aziz](http://wilkeraziz.github.io) at the University of Amsterdam.