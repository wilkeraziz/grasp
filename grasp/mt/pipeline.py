"""
:Authors: - Wilker Aziz
"""
import random
import os
import itertools
import numpy as np

import grasp.ptypes as ptypes

import grasp.semiring as semiring

from grasp.loss.fast_bleu import DecodingBLEU

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
from grasp.formal.traversal import bracketed_string
from grasp.formal.traversal import yield_string

from grasp.recipes import dummyfunc
from grasp.recipes import traceit
from grasp.recipes import smart_ropen
from grasp.recipes import smart_wopen
from grasp.recipes import pickle_it
from grasp.recipes import unpickle_it

from grasp.scoring.lookup import RuleTable
from grasp.scoring.stateless import WordPenalty
from grasp.scoring.stateless import ArityPenalty
from grasp.scoring.lm import StatelessLM
from grasp.scoring.lm import KenLM
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
from grasp.alg.rescoring import SlicedRescoring
from grasp.alg.chain import apply_filters
from grasp.alg.chain import group_by_identity
from grasp.alg.chain import group_by_projection
from grasp.alg.value import acyclic_value_recursion


def is_step_complete(step, saving, redo):
    return step in saving and os.path.exists(saving[step]) and not redo


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



def load_feature_extractors(rt=None, wp=None, ap=None, slm=None, lm=None) -> 'tuple':
    """
    Load feature extractors depending on command line options.

    For now we have the following extractors:

        * RuleTable
        * WordPenalty
        * ArityPenalty
        * StatelessLM
        * KenLM

    :return: a tuple of Extractor objects
    """
    extractors = []

    if rt:
        extractor = RuleTable(uid=len(extractors),
                              name='RuleTable')
        extractors.append(extractor)

    if wp:
        extractor = WordPenalty(uid=len(extractors),
                                name=wp[0],
                                penalty=float(wp[1]))
        extractors.append(extractor)

    if ap:
        extractor = ArityPenalty(uid=len(extractors),
                                 name=ap[0],
                                 penalty=float(ap[1]))
        extractors.append(extractor)

    if slm:
        extractor = StatelessLM(uid=len(extractors),
                                name=slm[0],
                                order=int(slm[1]),
                                path=slm[2])
        extractors.append(extractor)

    if lm:
        extractor = KenLM(uid=len(extractors),
                          name=lm[0],
                          order=int(lm[1]),
                          path=lm[2])
        extractors.append(extractor)

    return tuple(extractors)


def load_model(description, weights, init):
    """

    :param description: path to Extractor constructors
    :param weights: path to weights
    :param init: initialisation strategy
    :return: ModelContainer
    """
    extractors = construct_extractors(description)
    if not weights and init is None:
        raise ValueError('Either provide a file containing weights or an initialisation strategy')
    if weights:
        wmap = read_weights(weights)
    else:
        if init == 'uniform':
            wmap = make_weight_map(extractors, InitialWeightFunction.uniform(len(extractors)))
        elif init == 'random':
            wmap = make_weight_map(extractors, InitialWeightFunction.normal())
        else:
            wmap = make_weight_map(extractors, InitialWeightFunction.constant(float(init)))
    return make_models(wmap, extractors)


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


def make_input_dfa(seg) -> 'DFA':
    """
    Create a DFA view of the input.
    """

    input_dfa = make_dfa(seg.src_tokens())
    return input_dfa


def make_reference_dfa(seg) -> 'DFA':
    return make_dfa_set([ref.split() for ref in seg.refs], semiring.inside.one)


def parse_dfa(hg, root, dfa, goal_rule, bottomup=True) -> 'Hypergraph':
    """
    Intersect a (possibly cyclic) hypergaph and a DFA.
    """
    #  2a. get a parser and intersect the source FSA
    if bottomup:
        parser = NederhofParser(hg, dfa, semiring.inside)
    else:
        parser = EarleyParser(hg, dfa, semiring.inside)
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


def pass0(seg, extra_grammar_paths=[], glue_grammar_paths=[], pass_through=True,
          default_symbol='X', goal_str='GOAL', start_str='S', n_goal=0,
          saving={}, redo=True, log=dummyfunc) -> 'Hypergraph':
    """
    Pass0 consists in parsing with the source side of the grammar.
    For now, pass0 does not do any scoring (not even local), but it could (TODO).

    Steps
        1. Make a hypergraph view of the grammar
        2. Make an input DFA
        3. Parse the input DFA

    :return: source forest
    """
    if is_step_complete('forest', saving, redo):
        return unpickle_it(saving['forest'])

    # here we need to decode for sure
    log('[%d] Make hypergraph view of all available grammars', seg.id)
    # make a hypergraph view of all available grammars
    grammar = make_grammar_hypergraph(seg,
                                      extra_grammar_paths=extra_grammar_paths,
                                      glue_grammar_paths=glue_grammar_paths,
                                      pass_through=pass_through,
                                      default_symbol=default_symbol)

    # parse source lattice
    log('[%d] Parse source DFA', seg.id)
    goal_maker = GoalRuleMaker(goal_str=goal_str, start_str=start_str, n=n_goal)
    dfa = make_input_dfa(seg)
    forest = parse_dfa(grammar,
                       grammar.fetch(Nonterminal(start_str)),
                       dfa,
                       goal_maker.get_iview(),
                       bottomup=True)
    if 'forest' in saving:
        pickle_it(saving['forest'], forest)
    return forest


def pass1(seg, src_forest, model,
          saving={}, redo=True,
          log=dummyfunc) -> 'str':
    """
    Pass1 consists in obtaining a target forest and locally scoring it.

    Steps
        1. Project target side of the forest
        2. Lookup scoring
        3. Stateless scoring

    :return: source forest
    """

    if is_step_complete('forest', saving, redo):
        tgt_forest = unpickle_it(saving['forest'])
    else:
        # target projection
        log('[%d] Project target rules', seg.id)
        tgt_forest = make_target_forest(src_forest)
        if 'forest' in saving:
            pickle_it(saving['forest'], tgt_forest)

    # local scoring
    if is_step_complete('lookup', saving, redo):
        lookup_comps = unpickle_it(saving['lookup'])
    else:
        log('[%d] Lookup scoring', seg.id)
        lookup_comps = get_lookup_components(tgt_forest, model.lookup.extractors())
        if 'lookup' in saving:
            pickle_it(saving['lookup'], lookup_comps)

    if is_step_complete('stateless', saving, redo):
        stateless_comps = unpickle_it(saving['stateless'])
    else:
        log('[%d] Stateless scoring', seg.id)
        stateless_comps = get_stateless_components(tgt_forest, model.stateless.extractors())
        if 'stateless' in saving:
            pickle_it(saving['stateless'], stateless_comps)

    return tgt_forest, lookup_comps, stateless_comps


def pass2(seg, forest,
          lookup_scorer, stateless_scorer, stateful_scorer,
          goal_rule, omega=None,
          saving={}, redo=True, log=dummyfunc) -> 'tuple':
    """
    Pass2 consists in exactly rescoring a forest.
    :return: rescored forest (a Hypergraph), and components (one FComponents object per edge)
    """

    if is_step_complete('forest', saving, redo) and is_step_complete('components', saving, redo) :
        rescored_forest = unpickle_it(saving['forest'])
        components = unpickle_it(saving['components'])
        return rescored_forest, components

    log('[%d] Forest rescoring', seg.id)
    rescored_forest, components = rescore_forest(forest,
                                                 0,
                                                 lookup_scorer,
                                                 stateless_scorer,
                                                 stateful_scorer,
                                                 goal_rule=goal_rule,
                                                 omega=omega,
                                                 keep_components=True)
    if 'forest' in saving:
        pickle_it(saving['forest'], rescored_forest)
    if 'components' in saving:
        pickle_it(saving['components'], components)

    return rescored_forest, components


def draw_samples(forest,
                 omega,
                 tsort,
                 lookup_scorer,
                 stateless_scorer,
                 stateful_scorer,
                 n_samples, batch_size, within, initial, prior, burn, lag, temperature0,
                 goal_rule,
                 dead_rule):

    sampler = SlicedRescoring(forest,
                              omega,
                              tsort,
                              lookup_scorer,
                              stateless_scorer,
                              stateful_scorer,
                              semiring.inside,
                              goal_rule,
                              dead_rule)

    # here samples are represented as sequences of edge ids
    d0, markov_chain = sampler.sample(n_samples=n_samples,
                                      batch_size=batch_size,
                                      within=within,
                                      initial=initial,
                                      prior=prior,
                                      burn=burn,
                                      lag=lag,
                                      temperature0=temperature0)

    return d0, markov_chain


def consensus(seg, forest, samples, log=dummyfunc):
    # total number of samples kept
    n_samples = len(samples)
    projections = group_by_projection(samples, lambda d: yield_string(forest, d.edges))

    log('[%d] Consensus decoding', seg.id)
    # translation strings
    support = [group.key for group in projections]
    # empirical distribution
    posterior = np.array([float(group.count) / n_samples for group in projections], dtype=ptypes.weight)
    # consensus decoding
    scorer = DecodingBLEU(support, posterior)
    losses = np.array([scorer.loss(y) for y in support], dtype=ptypes.weight)
    # order samples by least loss, then by max prob
    ranking = sorted(range(len(support)), key=lambda i: (losses[i], -posterior[i]))
    return [(losses[i], posterior[i], support[i]) for i in ranking]


def make_slice_sampler(seg, model,
                       extra_grammar_paths=[], glue_grammar_paths=[], pass_through=True,
                       default_symbol='X', goal_str='GOAL', start_str='S',
                       saving={}, redo=True,
                       log=dummyfunc) -> 'str':
    """
    Return the best translation according to a consensus decision rule.
    :return: best translation string
    """

    # check for pass1
    if all(is_step_complete(step, saving, redo) for step in ['forest', 'lookup', 'stateless']):
        tgt_forest = unpickle_it(saving['forest'])
        lookup_comps = unpickle_it(saving['lookup'])
        stateless_comps = unpickle_it(saving['stateless'])
    else:
        src_forest = pass0(seg,
                           extra_grammar_paths=extra_grammar_paths,
                           glue_grammar_paths=glue_grammar_paths,
                           pass_through=pass_through,
                           default_symbol=default_symbol,
                           goal_str=goal_str,
                           start_str=start_str,
                           n_goal=0,
                           saving={},
                           redo=redo,
                           log=log)

        # pass1: local scoring
        tgt_forest, lookup_comps, stateless_comps = pass1(seg,
                                                          src_forest,
                                                          model,
                                                          saving=saving,
                                                          redo=redo,
                                                          log=log)

    # l(d)
    lfunc = TableLookupFunction(np.array([semiring.inside.times(model.lookup.score(ff1),
                                                                model.stateless.score(ff2))
                                          for ff1, ff2 in zip(lookup_comps, stateless_comps)], dtype=ptypes.weight))
    # top sort table
    tsort = AcyclicTopSortTable(tgt_forest)
    goal_maker = GoalRuleMaker(goal_str=goal_str, start_str=start_str, n=1)
    # slice sampler
    sampler = SlicedRescoring(tgt_forest,
                              lfunc,
                              tsort,
                              TableLookupScorer(model.dummy),
                              StatelessScorer(model.dummy),
                              StatefulScorer(model.stateful),
                              semiring.inside,
                              goal_rule=goal_maker.get_oview(),
                              dead_rule=make_dead_oview())
    return tgt_forest, lfunc, tsort, sampler


def decode(seg, args, n_samples, model, saving, redo, log=dummyfunc):

    # first we check whether the decisions have been completed before
    if is_step_complete('decisions', saving, redo):
        log('[%d] Reusing decisions', seg.id)
        with smart_ropen(saving['decisions']) as fi:
            for line in fi.readlines():
                if line.startswith('#'):
                    continue
                line = line.strip()
                if not line:
                    continue
                fields = line.split(' ||| ')  # that should be (loss, posterior, solution)
                if len(fields) == 3:
                    return fields[2]  # that's the solution

    forest, lfunc, tsort, sampler = make_slice_sampler(seg,
                                                       model,
                                                       extra_grammar_paths=args.extra_grammar,
                                                       glue_grammar_paths=args.glue_grammar,
                                                       pass_through=args.pass_through,
                                                       default_symbol=args.default_symbol,
                                                       goal_str=args.goal,
                                                       start_str=args.start,
                                                       saving=saving,
                                                       redo=args.redo,
                                                       log=log)

    d0, markov_chain = sampler.sample(n_samples=n_samples,
                                      batch_size=args.batch,
                                      within=args.within,
                                      initial=args.initial,
                                      prior=args.prior,
                                      burn=args.burn,
                                      lag=args.lag,
                                      temperature0=args.temperature0)

    # TODO: save stuff

    samples = apply_filters(markov_chain,
                            burn=args.burn,
                            lag=args.lag)

    decisions = consensus(seg, forest, samples)
    if 'decisions' in saving:
        # write all decisions to file
        with smart_wopen(saving['decisions']) as fo:
            print('# co-loss ||| posterior ||| solution', file=fo)
            for l, p, y in decisions:
                print('{0} ||| {1} ||| {2}'.format(l, p, y), file=fo)
    return decisions[0][2]  # return best translation


@traceit
def training_decode(seg, args, n_samples, staticdir, decisiondir, model, redo, log=dummyfunc):

    saving = {
        'forest': '{0}/{1}.hyp.forest'.format(staticdir, seg.id),
        'lookup': '{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id),
        'stateless': '{0}/{1}.hyp.ffs.stateless'.format(staticdir, seg.id),
        'decisions': '{0}/{1}.gz'.format(decisiondir, seg.id)
    }
    return decode(seg, args, n_samples, model, saving, redo, log)


@traceit
def training_biparse(seg, args, workingdir, model, log=dummyfunc) -> 'bool':
    """
    Steps:
        I. Pass0 and pass1: parse source, project, local scoring
        II. Pass2
            - make a reference DFA
            - parse the reference DFA
            - fully score the reference forest (lookup, stateless, stateful)
                - save rescored forest and components
    :return: whether or not the input is bi-parsable
    """

    pass1_files = ['{0}/{1}.hyp.forest'.format(workingdir, seg.id),
                   '{0}/{1}.hyp.ffs.rule'.format(workingdir, seg.id),
                   '{0}/{1}.hyp.ffs.stateless'.format(workingdir, seg.id)]
    ref_files = ['{0}/{1}.ref.ffs.all'.format(workingdir, seg.id),
                 '{0}/{1}.ref.forest'.format(workingdir, seg.id)]

    # check for redundant work
    if all(os.path.exists(path) for path in pass1_files) and not args.redo:
        if all(os.path.exists(path) for path in ref_files):
            log('[%d] Reusing forests for segment', seg.id)
            return True   # parsable
        else:
            return False  # not parsable

    # pass0: parsing

    src_forest = pass0(seg,
                       extra_grammar_paths=args.extra_grammar,
                       glue_grammar_paths=args.glue_grammar,
                       pass_through=args.pass_through,
                       default_symbol=args.default_symbol,
                       goal_str=args.goal,
                       start_str=args.start,
                       n_goal=0,
                       saving={},
                       redo=args.redo,
                       log=log)

    # pass1: local scoring

    saving1 = {
        'forest': '{0}/{1}.hyp.forest'.format(workingdir, seg.id),
        'lookup': '{0}/{1}.hyp.ffs.rule'.format(workingdir, seg.id),
        'stateless': '{0}/{1}.hyp.ffs.stateless'.format(workingdir, seg.id)
    }

    tgt_forest, lookup_comps, stateless_comps = pass1(seg,
                                                      src_forest,
                                                      model,
                                                      saving=saving1,
                                                      redo=args.redo,
                                                      log=log)


    # parse reference lattice
    log('[%d] Parse reference DFA', seg.id)
    ref_dfa = make_reference_dfa(seg)
    goal_maker = GoalRuleMaker(goal_str=args.goal, start_str=args.start, n=1)
    ref_forest = parse_dfa(tgt_forest,
                           0,
                           ref_dfa,
                           goal_maker.get_oview(),
                           bottomup=False)

    if not ref_forest:
        return False  # not parsable

    # pass2: rescore reference forest

    saving2 = {
        'forest': '{0}/{1}.ref.forest'.format(workingdir, seg.id),
        'components': '{0}/{1}.ref.ffs.all'.format(workingdir, seg.id)
    }
    goal_maker.update()
    pass2(seg, ref_forest,
          TableLookupScorer(model.lookup),
          StatelessScorer(model.stateless),
          StatefulScorer(model.stateful),
          goal_maker.get_oview(),
          saving=saving2, redo=args.redo,
          log=log)

    return True  # parsable


@traceit
def training_parse(seg, args, workingdir, model, log=dummyfunc) -> 'bool':
    """
    Steps:
        I. Pass0 and pass1: parse source, project, local scoring
        II. Pass2
            - make a reference DFA
            - parse the reference DFA
            - fully score the reference forest (lookup, stateless, stateful)
                - save rescored forest and components
    :return: whether or not the input is bi-parsable
    """

    pass1_files = ['{0}/{1}.hyp.forest'.format(workingdir, seg.id),
                   '{0}/{1}.hyp.ffs.rule'.format(workingdir, seg.id),
                   '{0}/{1}.hyp.ffs.stateless'.format(workingdir, seg.id)]

    # check for redundant work
    if all(os.path.exists(path) for path in pass1_files) and not args.redo:
        return True

    # pass0: parsing

    src_forest = pass0(seg,
                       extra_grammar_paths=args.extra_grammar,
                       glue_grammar_paths=args.glue_grammar,
                       pass_through=args.pass_through,
                       default_symbol=args.default_symbol,
                       goal_str=args.goal,
                       start_str=args.start,
                       n_goal=0,
                       saving={},
                       redo=args.redo,
                       log=log)
    if not src_forest:
        return False
    # pass1: local scoring

    saving1 = {
        'forest': '{0}/{1}.hyp.forest'.format(workingdir, seg.id),
        'lookup': '{0}/{1}.hyp.ffs.rule'.format(workingdir, seg.id),
        'stateless': '{0}/{1}.hyp.ffs.stateless'.format(workingdir, seg.id)
    }

    tgt_forest, lookup_comps, stateless_comps = pass1(seg,
                                                      src_forest,
                                                      model,
                                                      saving=saving1,
                                                      redo=args.redo,
                                                      log=log)

    return True
