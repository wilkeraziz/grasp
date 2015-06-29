"""
:Authors: - Wilker Aziz
"""
import re
from os.path import isfile


class SegmentMetaData(object):
    """
    A simple container for input segments
    """

    def __init__(self, sid, src, grammar, refs=[]):
        self.sid_ = sid
        self.grammar_ = grammar
        self.src_ = src
        self.refs_ = tuple(refs)

    @property
    def id(self):
        return self.sid_

    @property
    def src(self):
        return self.src_

    def src_tokens(self, boundaries=False):
        if not boundaries:
            return self.src.split()
        else:
            return ['<s>'] + self.src.split() + ['</s>']

    @property
    def refs(self):
        return self.refs_

    @property
    def grammar(self):
        return self.grammar_

    def __str__(self):
        return 'grammar=%s\tsrc=%s' % (self.grammar_, self.src_)

    def to_sgm(self, dump_refs=True):
        if dump_refs and self.refs_:
            return '<seg grammar="{1}" id="{0}">{2}</seg> ||| {3}'.format(self.sid_,
                                                                          self.grammar_,
                                                                          self.src_,
                                                                          ' ||| '.join(str(ref) for ref in self.refs_))
        else:
            return '<seg grammar="{1}" id="{0}">{2}</seg>'.format(self.sid_,
                                                                  self.grammar_,
                                                                  self.src_)

    @staticmethod
    def parse(line, sid=None, grammar_dir=None, mode='cdec-sgml'):
        if mode == 'cdec-sgml':
            args = parse_cdec_sgml(line)
        else:
            raise Exception('unknown input format: %s' % mode)
        # overrides sentence id
        if sid is not None:
            args['sid'] = sid
        else:
            sid = args['sid']
        # overrides grammar
        if grammar_dir is not None:
            args['grammar'] = '{0}/grammar.{1}.gz'.format(grammar_dir, args['sid'])
        else:
            grammar_dir = args['grammar']
        # sanity checks
        if not isfile(args['grammar']):
            raise FileNotFoundError('Grammar file not found: %s' % args['grammar'])
        # construct segment
        return SegmentMetaData(**args)


def parse_cdec_sgml(sgml_str):
    """
    Parse a cdec-style sgml-formatted line.
    :param sgml_str:
    :return: dict with grammar-path, id, input and references
    """
    parts = sgml_str.split(' ||| ')
    if not parts:
        raise Exception('Missing fields' % sgml_str)
    pattern = re.compile(r'<seg grammar="([^"]+)" id="([0-9]+)">(.+)</seg>')
    match = pattern.match(parts[0])
    if match is None:
        raise SyntaxError('Bad sgml: %s' % parts[0])
    groups = match.groups()
    return {'grammar': groups[0],
            'sid': groups[1],
            'src': groups[2],
            'refs': [ref.strip() for ref in parts[1:]]}
