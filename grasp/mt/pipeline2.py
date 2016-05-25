"""
:Authors: - Wilker Aziz
"""
import random
import os
import itertools
import numpy as np
import sys
from collections import defaultdict

import grasp.ptypes as ptypes

import grasp.semiring as semiring

from types import SimpleNamespace

from grasp.mt.segment import SegmentMetaData
import grasp.mt.cdec_format as cdeclib
from grasp.mt.input import make_pass_grammar
from grasp.mt.util import GoalRuleMaker
from grasp.mt.util import make_dead_oview

import grasp.formal.scfgop as scfgop
from grasp.formal.hg import Hypergraph
from grasp.formal.fsa import make_dfa
from grasp.formal.fsa import make_dfa_set
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.wfunc import TableLookupFunction
from grasp.formal.wfunc import ConstantFunction
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
from grasp.scoring.model import DummyModel, Model, ModelView, ModelContainer
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


def prefixlog(prefix, log):
    return lambda x: log(prefix, x)


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
    segments = [SegmentMetaData.parse(input_str, grammar_dir=grammar_dir)
                for input_str in istream]
    if shuffle:
        random.shuffle(segments)
    return segments


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


def get_factorised_models(model: Model, path='') -> (ModelView, ModelView):
    """
    Return a joint and a conditional factorisation of the model.

    :param model: a Model
    :param path: (optional) path to a file changing the default way of factorising a model
    :return: joint view and conditional view
    """
    joint_changes = defaultdict(set)
    conditional_changes = defaultdict(set)
    if path:
        with smart_ropen(path) as fi:
            changes = None
            for line in fi:
                line = line.strip()
                if not line or line.startswith('#'):  # ignore comments and empty lines
                    continue
                if line == '[joint]':
                    changes = joint_changes
                elif line == '[conditional]':
                    changes = conditional_changes
                elif changes is None:
                    raise ValueError('Syntax error in factorisation file')
                elif line.startswith('local='):
                    name = line.replace('local=', '', 1)
                    changes['local'].add(name)
                elif line.startswith('nonlocal='):
                    name = line.replace('nonlocal=', '', 1)
                    changes['nonlocal'].add(name)

    joint_model = ModelView(model.wmap, model.extractors(),
                            local_names=joint_changes['local'],
                            nonlocal_names=joint_changes['nonlocal'])
    conditional_model = ModelView(model.wmap, model.extractors(),
                                  local_names=conditional_changes['local'],
                                  nonlocal_names=conditional_changes['nonlocal'])
    return joint_model, conditional_model


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


def preprocessed_training_files(stem):
    return {'joint.forest': '{0}.joint.forest'.format(stem),
            'joint.components': '{0}.joint.components'.format(stem),
            'conditional.forest': '{0}.conditional.forest'.format(stem),
            'conditional.components': '{0}.conditional.components'.format(stem)}


def exact_rescoring(model: ModelContainer,
                    forest: Hypergraph, goal_maker: GoalRuleMaker, log=dummyfunc) -> SimpleNamespace:
    """
    Exactly rescore a forest with a certain model.

    :param model: an instance of ModelContainer
    :param forest: a Hypergraph
    :param goal_maker: an object to deliver (output view of) goal rules
    :param log: a logging function
    :return: result.forest and result.components as a SimpleNamespace object
    """
    result = SimpleNamespace()

    if not model.stateful:  # when the model is not stateful, we don't need Earley
        log('Lookup scoring')
        lookup_comps = get_lookup_components(forest, model.lookup.extractors())  # lookup
        log('Stateless scoring')
        stateless_comps = get_stateless_components(forest, model.stateless.extractors())  # stateless
        result.forest = forest
        result.components = [FComponents([comps1, comps2]) for comps1, comps2 in zip(lookup_comps, stateless_comps)]

    else:  # here we cannot avoid it
        log('Forest rescoring')
        goal_maker.update()
        result.forest, result.components = rescore_forest(forest,
                                                          0,
                                                          TableLookupScorer(model.lookup),
                                                          StatelessScorer(model.stateless),
                                                          StatefulScorer(model.stateful),
                                                          goal_rule=goal_maker.get_oview(),
                                                          keep_components=True)

    return result


def biparse(seg: SegmentMetaData, options: SimpleNamespace,
            joint_model: ModelView, conditional_model: ModelView,
            workingdir=None, redo=True, log=dummyfunc) -> SimpleNamespace:
    """
    Biparse a segment using a local model.
    1. we parse the source with a joint model
    2. we bi-parse source and target with a conditional model
    This separation allows us to factorise these models differently wrt local/nonlocal components.
    For example, an LM maybe seen as a local (read tractable) component of a conditional model,
     and as a nonlocal (read intractable) component of a joint model.
    An implementation detail: bi-parsing is implemented as a cascade of intersections (with projections in between).

    :param seg: a segment
    :param options: parsing options
    :param joint_model: a factorised view of the joint model, here we use only the local components
    :param conditional_model: a factorised view of the conditional, here we use only the local components
    :param workingdir: where to save files
    :param redo: whether or not previously saved computation should be discarded
    :param log: a logging function
    :return: result.{joint,conditional}.{forest,components} for the respective local model
    """

    if workingdir:
        saving = preprocessed_training_files('{0}/{1}'.format(workingdir, seg.id))
    else:
        saving = {}

    steps = ['joint.forest', 'joint.components', 'conditional.forest', 'conditional.components']
    result = SimpleNamespace()
    result.joint = SimpleNamespace()
    result.conditional = SimpleNamespace()

    if all(is_step_complete(step, saving, redo) for step in steps):
        log('[%d] Reusing joint and conditional distributions from files', seg.id)
        result.joint.forest = unpickle_it(saving['joint.forest'])
        result.joint.components = unpickle_it(saving['joint.components'])
        result.conditional.forest = unpickle_it(saving['conditional.forest'])
        result.conditional.components = unpickle_it(saving['conditional.components'])
        return result

    # 1. Make a grammar

    # here we need to decode for sure
    log('[%d] Make hypergraph view of all available grammars', seg.id)
    # make a hypergraph view of all available grammars
    grammar = make_grammar_hypergraph(seg,
                                      extra_grammar_paths=options.extra_grammars,
                                      glue_grammar_paths=options.glue_grammars,
                                      pass_through=options.pass_through,
                                      default_symbol=options.default_symbol)
    #print('GRAMMAR')
    #print(grammar)

    # 2. Joint distribution - Step 1: parse source lattice
    n_goal = 0
    log('[%d] Parse source DFA', seg.id)
    goal_maker = GoalRuleMaker(goal_str=options.goal, start_str=options.start, n=n_goal)
    src_dfa = make_input_dfa(seg)
    src_forest = parse_dfa(grammar,
                           grammar.fetch(Nonterminal(options.start)),
                           src_dfa,
                           goal_maker.get_iview(),
                           bottomup=True,
                           constraint=HieroConstraints(grammar, src_dfa, options.max_span))
    #print('SOURCE')
    #print(src_forest)

    if not src_forest:
        raise ValueError('I cannot parse the input lattice: i) make sure your grammar has glue rules; ii) make sure it handles OOVs')

    # 3. Target projection of the forest
    log('[%d] Project target rules', seg.id)
    tgt_forest = make_target_forest(src_forest)
    #print('TARGET')
    #print(tgt_forest)

    # 4. Joint distribution - Step 2: scoring

    log('[%d] Joint model: (exact) local scoring', seg.id)
    result.joint = exact_rescoring(joint_model.local_model(), tgt_forest, goal_maker, log)

    # save joint distribution
    if 'joint.forest' in saving:
        pickle_it(saving['joint.forest'], result.joint.forest)
    if 'joint.components' in saving:
        pickle_it(saving['joint.components'], result.joint.components)

    #print('JOINT')
    #print(result.joint.forest)

    # 5. Conditional distribution - Step 1: parse the reference lattice

    log('[%d] Parse reference DFA', seg.id)
    ref_dfa = make_reference_dfa(seg)
    goal_maker.update()
    ref_forest = parse_dfa(result.joint.forest,
                           0,
                           ref_dfa,
                           goal_maker.get_oview(),
                           bottomup=False)

    if not ref_forest:  # reference cannot be parsed
        log('[%d] References cannot be parsed', seg.id)
        result.conditional.forest = ref_forest
        result.conditional.components = []
    else:
        # 6. Conditional distribution - Step 2: scoring
        log('[%d] Conditional model: exact (local) scoring', seg.id)
        result.conditional = exact_rescoring(conditional_model.local_model(), ref_forest, goal_maker, log)

    # save conditional distribution
    if 'conditional.forest' in saving:
        pickle_it(saving['conditional.forest'], result.conditional.forest)
    if 'conditional.components' in saving:
        pickle_it(saving['conditional.components'], result.conditional.components)

    return result
