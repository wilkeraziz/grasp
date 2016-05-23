"""
:Authors: - Wilker Aziz
"""
import random
import os
import itertools
import numpy as np
import sys

import grasp.ptypes as ptypes

import grasp.semiring as semiring

from grasp.mt.segment import SegmentMetaData
import grasp.mt.cdec_format as cdeclib
from grasp.mt.input import make_pass_grammar
from grasp.mt.util import GoalRuleMaker
from grasp.mt.util import make_dead_oview

import grasp.formal.scfgop as scfgop
from grasp.formal.fsa import make_dfa
from grasp.formal.fsa import make_dfa_set
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.wfunc import TableLookupFunction
from grasp.formal.wfunc import derivation_weight
from grasp.formal.traversal import bracketed_string
from grasp.formal.traversal import yield_string

from grasp.recipes import symlink
from grasp.recipes import dummyfunc
from grasp.recipes import smart_ropen
from grasp.recipes import smart_wopen
from grasp.recipes import pickle_it
from grasp.recipes import unpickle_it

from grasp.scoring.frepr import FComponents
from grasp.scoring.scorer import TableLookupScorer
from grasp.scoring.scorer import StatelessScorer
from grasp.scoring.scorer import StatefulScorer
from grasp.scoring.model import DummyModel
from grasp.scoring.util import make_weight_map
from grasp.scoring.util import InitialWeightFunction
from grasp.scoring.util import construct_extractors
from grasp.scoring.util import read_weights
from grasp.scoring.util import make_models

from grasp.cfg.model import DummyConstant
from grasp.cfg.symbol import Nonterminal

from grasp.alg.deduction import NederhofParser
from grasp.alg.deduction import EarleyParser
from grasp.alg.deduction import EarleyRescorer
from grasp.alg.rescoring import score_derivation
from grasp.alg.chain import group_by_identity
from grasp.alg.chain import group_by_projection
from grasp.alg.inference import AncestralSampler
from grasp.alg.impsamp import ISDerivation, ISYield
from grasp.alg.constraint import Constraint as DummyConstraint
from grasp.alg.constraint import HieroConstraints


def is_step_complete(step, saving, redo):
    return step in saving and os.path.exists(saving[step]) and not redo


def all_steps_complete(saving, redo):
    return all(is_step_complete(step, saving, redo) for step in saving.keys())


def read_segments_from_stream(istream, grammar_dir=None, shuffle=False) -> 'tuple':
    """
    Read cdec-formated input segments (possibly along with their reference translations) from an input stream.
    :param istream: input stream
    :param grammar_dir: overwrites grammar directory
    :param shuffle: shuffle segments inplace
    :return: tuple of SegmentMetaData objects
    """
    if shuffle:
        segments = [SegmentMetaData.parse(input_str, grammar_dir=grammar_dir)
                    for input_str in istream]
        random.shuffle(segments)
        return tuple(segments)
    else:
        return tuple(SegmentMetaData.parse(input_str, grammar_dir=grammar_dir)
                     for input_str in istream)


def read_segments_from_file(path, grammar_dir=None, shuffle=False) -> 'tuple':
    """
    Read cdec-formated input segments (possibly along with their reference translations) from a file.
    :param path: path to file (possibly gzipped)
    :param grammar_dir: overwrites grammar directory
    :param shuffle: shuffle segments inplace
    :return: tuple of SegmentMetaData objects
    """
    return read_segments_from_stream(smart_ropen(path), grammar_dir=grammar_dir, shuffle=shuffle)


def save_segments(path, segments):
    with smart_wopen(path) as fo:
        for seg in segments:
            print(seg.to_sgm(True), file=fo)


def save_references(path, segments):
    with smart_wopen(path) as fo:
        for seg in segments:
            print(' ||| '.join(seg.refs), file=fo)


def load_model(description, weights, init, temperature=1.0):
    """

    :param description: path to Extractor constructors
    :param weights: path to weights
    :param init: initialisation strategy
    :param temperature: scale the model
    :return: ModelContainer
    """
    extractors = construct_extractors(description)
    if not weights and init is None:
        raise ValueError('Either provide a file containing weights or an initialisation strategy')
    if weights:
        wmap = read_weights(weights, temperature=temperature)
    else:
        if init == 'uniform':
            wmap = make_weight_map(extractors, InitialWeightFunction.uniform(len(extractors)))
        elif init == 'random':
            wmap = make_weight_map(extractors, InitialWeightFunction.normal())
        else:
            wmap = make_weight_map(extractors, InitialWeightFunction.constant(float(init)))
    return make_models(wmap, extractors)


def make_input_dfa(seg) -> 'DFA':
    """
    Create a DFA view of the input.
    """

    input_dfa = make_dfa(seg.src_tokens())
    return input_dfa


def make_reference_dfa(seg) -> 'DFA':
    return make_dfa_set([ref.split() for ref in seg.refs], semiring.inside.one)


def parse_dfa(hg, root, dfa, goal_rule, bottomup=True, constraint=DummyConstraint()) -> 'Hypergraph':
    """
    Intersect a (possibly cyclic) hypergaph and a DFA.
    """
    #  2a. get a parser and intersect the source FSA
    if bottomup:
        parser = NederhofParser(hg, dfa, semiring.inside, constraint=constraint)
    else:
        parser = EarleyParser(hg, dfa, semiring.inside, constraint=constraint)
    return parser.do(root,goal_rule)


def make_target_forest(source_forest, rulescorer=TableLookupScorer(DummyModel())) -> 'Hypergraph':
    return scfgop.output_projection(source_forest, semiring.inside, rulescorer)


def get_lookup_components(forest, lookup_extractors) -> 'list':
    """
    Return the TableLookup representation of each edge in the forest.
    """
    return scfgop.lookup_components(forest, lookup_extractors)


def get_stateless_components(forest, stateless_extractors) -> 'list':
    """
    Return the Stateless representation of each edge in the forest.
    """
    return scfgop.stateless_components(forest, stateless_extractors)


def rescore_forest(forest, root, lookup, stateless, stateful, goal_rule, omega=None, keep_components=True) -> 'tuple':
    """
    Return a rescored forest and a list of component vectors.
    """

    rescorer = EarleyRescorer(forest,
                              lookup,
                              stateless,
                              stateful,
                              semiring.inside,
                              omega=omega,
                              map_edges=False,
                              keep_frepr=keep_components)
    rescored_forest = rescorer.do(root, goal_rule)
    return rescored_forest, rescorer.components()


def make_grammar_hypergraph(seg, extra_grammar_paths=[],
                            glue_grammar_paths=[],
                            pass_through=True,
                            default_symbol='X') -> 'Hypergraph':
    """
    Load grammars (i.e. main, extra, glue, passthrough) and prepare input FSA.
    :return: Hypergraph grammar
    """
    # 1. Load grammars
    #  1a. additional grammars
    extra_grammars = []
    if extra_grammar_paths:
        for grammar_path in extra_grammar_paths:
            grammar = cdeclib.load_grammar(grammar_path)
            extra_grammars.append(grammar)
    #  1b. glue grammars
    glue_grammars = []
    if glue_grammar_paths:
        for glue_path in glue_grammar_paths:
            glue = cdeclib.load_grammar(glue_path)
            glue_grammars.append(glue)
    #  1c. main grammar
    main_grammar = cdeclib.load_grammar(seg.grammar)

    # 2. Make a pass-through grammar for the given segment
    #  2a. pass-through grammar
    _, pass_grammar = make_pass_grammar(seg,
                                        list(itertools.chain([main_grammar], extra_grammars, glue_grammars)),
                                        semiring.inside,
                                        default_symbol)

    #  3a. put all (normal) grammars together
    if pass_through:
        grammars = list(itertools.chain([main_grammar], extra_grammars, [pass_grammar]))
    else:
        grammars = list(itertools.chain([main_grammar], extra_grammars))

    # and finally create a hypergraph based on the source side of the grammar
    # TODO: allow different models (other than DummyConstant)
    hg = scfgop.make_hypergraph_from_input_view(grammars,
                                                glue_grammars,
                                                DummyConstant(semiring.inside.one))
    return hg


def pass0(seg, options, n_goal=0, saving={}, redo=True, log=dummyfunc) -> 'Hypergraph':
    """
    Pass0 consists in parsing with the source side of the grammar.
    For now, pass0 does not do any scoring (not even local), but it could (TODO).

    Steps
        1. Make a hypergraph view of the grammar
        2. Make an input DFA
        3. Parse the input DFA

    :return: source forest
    """
    if is_step_complete('pass0.forest', saving, redo):
        return unpickle_it(saving['pass0.forest'])

    # here we need to decode for sure
    log('[%d] Make hypergraph view of all available grammars', seg.id)
    # make a hypergraph view of all available grammars
    grammar = make_grammar_hypergraph(seg,
                                      extra_grammar_paths=options.extra_grammars,
                                      glue_grammar_paths=options.glue_grammars,
                                      pass_through=options.pass_through,
                                      default_symbol=options.default_symbol)

    # parse source lattice
    log('[%d] Parse source DFA', seg.id)
    goal_maker = GoalRuleMaker(goal_str=options.goal, start_str=options.start, n=n_goal)
    dfa = make_input_dfa(seg)
    forest = parse_dfa(grammar,
                       grammar.fetch(Nonterminal(options.start)),
                       dfa,
                       goal_maker.get_iview(),
                       bottomup=True,
                       constraint=HieroConstraints(grammar, dfa, options.max_span))
    if 'pass0.forest' in saving:
        pickle_it(saving['pass0.forest'], forest)
    return forest


def pass0_to_pass1(seg, options, lookup, stateless, saving={}, redo=True, log=dummyfunc) -> 'str':
    """
    Pass1 consists in obtaining a target forest and locally scoring it.

    Steps
        1. Project target side of the forest
        2. Lookup scoring
        3. Stateless scoring

    :return: source forest
    """

    if is_step_complete('pass1.forest', saving, redo):  # try to reuse previous results
        tgt_forest = unpickle_it(saving['pass1.forest'])
    else: # execute pass0
        src_forest = pass0(seg, options, n_goal=0, saving={}, redo=redo, log=dummyfunc)
        # target projection
        log('[%d] Project target rules', seg.id)
        tgt_forest = make_target_forest(src_forest)
        if 'pass1.forest' in saving:
            pickle_it(saving['pass1.forest'], tgt_forest)

    if is_step_complete('pass1.components', saving, redo):
        components = unpickle_it(saving['pass1.components'])
    else:
        log('[%d] Lookup scoring', seg.id)
        lookup_comps = get_lookup_components(tgt_forest, lookup.extractors())
        log('[%d] Stateless scoring', seg.id)
        stateless_comps = get_stateless_components(tgt_forest, stateless.extractors())
        components = [FComponents([comps1, comps2]) for comps1, comps2 in zip(lookup_comps, stateless_comps)]
        if 'pass1.components' in saving:
            pickle_it(saving['pass1.components'], components)

    return tgt_forest, components


def pass0_to_pass2(seg, options, lookup, stateless, stateful, saving={}, redo=True, log=dummyfunc) -> 'tuple':
    """
    Pass2 consists in exactly rescoring a forest.
    :return: rescored forest (a Hypergraph), and components (one FComponents object per edge)
    """

    # We try to reuse previous results
    if is_step_complete('pass2.forest', saving, redo) and is_step_complete('pass2.components', saving, redo):
        forest = unpickle_it(saving['pass2.forest'])
        components = unpickle_it(saving['pass2.components'])
        return forest, components

    # We check whether we need pass2
    if not stateful:  # execute passes 0 to 1 only
        forest, components = pass0_to_pass1(seg,
                                            options,
                                            lookup,
                                            stateless,
                                            saving,
                                            redo=redo,
                                            log=log)

        # TODO: complete components with empty stateful model
        # save (or link) forest
        if 'pass2.forest' in saving:
            if 'pass1.forest' in saving:
                symlink(saving['pass1.forest'], saving['pass2.forest'])
            else:
                pickle_it(saving['pass2.forest'], forest)
        # save (or link) components
        if 'pass2.components' in saving:
            if 'pass1.components' in saving:
                symlink(saving['pass1.components'], saving['pass2.components'])
            else:
                pickle_it(saving['pass2.components'], components)
        return forest, components

    # From here we are sure we have stateful scorers
    # then we first execute passes 0 to 1 (and discard dummy components)
    forest, _ = pass0_to_pass1(seg,
                               options,
                               TableLookupScorer(DummyModel()),
                               StatelessScorer(DummyModel()),
                               saving,
                               redo=redo,
                               log=log)

    # then we fully re-score the forest (keeping all components)
    log('[%d] Forest rescoring', seg.id)
    goal_maker = GoalRuleMaker(goal_str=options.goal, start_str=options.start, n=1)
    forest, components = rescore_forest(forest,
                                        0,
                                        TableLookupScorer(lookup),
                                        StatelessScorer(stateless),
                                        StatefulScorer(stateful),
                                        goal_rule=goal_maker.get_oview(),
                                        keep_components=True)
    # save the forest
    if 'pass2.forest' in saving:
        pickle_it(saving['pass2.forest'], forest)
    # save the components
    if 'pass2.components' in saving:
        pickle_it(saving['pass2.components'], components)

    return forest, components


def importance_sample(seg, options, proxy, target, saving={}, redo=True, log=dummyfunc):
    """

    :param seg:
    :param options:
    :param proxy:
    :param target:
    :param log:
    :return:
    """

    if is_step_complete('is.samples', saving, redo):
        return unpickle_it(saving['is.samples'])

    q_forest, q_components = pass0_to_pass2(seg, options,
                                            TableLookupScorer(proxy.lookup),
                                            StatelessScorer(proxy.stateless),
                                            StatefulScorer(proxy.stateful),
                                            saving=saving,
                                            redo=redo,
                                            log=log)

    # Make unnormalised q(d)
    q_func = TableLookupFunction(np.array([proxy.score(comps) for comps in q_components], dtype=ptypes.weight))

    log('[%d] Q-forest: nodes=%d edges=%d', seg.id, q_forest.n_nodes(), q_forest.n_edges())
    tsort = AcyclicTopSortTable(q_forest)

    sampler = AncestralSampler(q_forest, tsort, omega=q_func)
    samples = sampler.sample(options.samples)

    d_groups = group_by_identity(samples)
    y_groups = group_by_projection(d_groups, lambda group: yield_string(q_forest, group.key))

    is_yields = []
    for y_group in y_groups:
        y = y_group.key
        is_derivations = []
        for d_group in y_group.values:
            edges = d_group.key
            # reduce q weights through inside.times
            q_score = derivation_weight(q_forest, edges, semiring.inside, omega=q_func)
            # reduce q components through inside.times
            q_comps = proxy.constant(semiring.inside.one)
            for e in edges:
                q_comps = q_comps.hadamard(q_components[e], semiring.inside.times)
            # compute p components and p score
            p_comps, p_score = score_derivation(q_forest, edges, semiring.inside,
                                                TableLookupScorer(target.lookup),
                                                StatelessScorer(target.stateless),
                                                StatefulScorer(target.stateful))
            # TODO: save {y => {edges: (q_comps, p_comps, count)}}
            is_derivations.append(ISDerivation(edges, q_comps, p_comps, d_group.count))
        is_yields.append(ISYield(y, is_derivations, y_group.count))
    if 'is.samples' in saving:
        pickle_it(saving['is.samples'], is_yields)

    return is_yields
