"""
Reports information about grammars.

:Authors: - Wilker Aziz
"""
import argparse
import logging
from grasp.cfg.reader import load_grammar
from tabulate import tabulate
from grasp.formal.hg import cfg_to_hg
from grasp.formal.topsort import LazyTopSortTable


def report(args):

    general_header = ['path', 'type', 'terminals', 'nonterminals', 'rules']
    general_info = []

    # Load main grammars
    logging.info('Loading main grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d',
                 cfg.n_terminals(),
                 cfg.n_nonterminals(),
                 len(cfg))
    general_info.append([args.grammar, 'main', cfg.n_terminals(), cfg.n_nonterminals(), len(cfg)])

    # Load additional grammars
    main_grammars = [cfg]
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('Loading additional grammar: %s', grammar_path)
            grammar = load_grammar(grammar_path, args.grammarfmt, args.log)
            logging.info('Additional grammar: terminals=%d nonterminals=%d productions=%d',
                         grammar.n_terminals(),
                         grammar.n_nonterminals(),
                         len(grammar))
            main_grammars.append(grammar)
            general_info.append([grammar_path, 'extra', grammar.n_terminals(), grammar.n_nonterminals(), len(grammar)])

    # Load glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('Loading glue grammar: %s', glue_path)
            glue = load_grammar(glue_path, args.grammarfmt, args.log)
            logging.info('Glue grammar: terminals=%d nonterminals=%d productions=%d', glue.n_terminals(),
                         glue.n_nonterminals(), len(glue))
            glue_grammars.append(glue)
            general_info.append([glue_path, 'glue', glue.n_terminals(), glue.n_nonterminals(), len(glue)])

    print(tabulate(general_info, general_header))

    hg = cfg_to_hg(main_grammars, glue_grammars)
    tsorter = LazyTopSortTable(hg, acyclic=False)

    if args.tsort:
        tsort = tsorter.do()
        with open('{0}.tsort'.format(args.output), 'w') as fo:
            print(tsort.pp(), file=fo)


def argparser():

    parser = argparse.ArgumentParser(prog='grasp.cfg.info',
                                     usage='python -m %(prog)s [options]',
                                     description="Output information about a (collection of) grammar(s)",
                                     epilog="")

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
                        type=str,
                        help='grammar file (or prefix to grammar files)')

    parser.add_argument('output',
                        type=str,
                        help='output prefix')

    parser.add_argument('--log',
                       action='store_true',
                       help='apply the log transform to the grammar (by default we assume this has already been done)')

    parser.add_argument('--grammarfmt',
                       type=str, default='bar', metavar='FMT',
                       choices=['bar', 'discodop', 'cdec'],
                       help="grammar format: bar, discodop, cdec")

    parser.add_argument('--extra-grammar',
                       action='append', default=[], metavar='PATH',
                       help="path to an additional grammar (multiple allowed)")
    parser.add_argument('--glue-grammar',
                       action='append', default=[], metavar='PATH',
                       help="glue rules are only applied to initial states (multiple allowed)")
    parser.add_argument('--tsort',
                       action='store_true',
                       help='topsort the grammar and report the partial ordering of its symbols')
    parser.add_argument('--verbose', '-v',
                       action='count', default=0,
                       help='increase the verbosity level')
    return parser


def configure():
    """
    Parse command line arguments, configures the main logger.
    :returns: command line arguments
    """

    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():
    """
    Configures the parser by parsing command line arguments and calling the core code.
    It might also profile the run if the user chose to do so.
    """

    args = configure()

    # Prepare output directories

    logging.info('Writing files to: %s.*', args.output)
    report(args)


if __name__ == '__main__':
    main()
