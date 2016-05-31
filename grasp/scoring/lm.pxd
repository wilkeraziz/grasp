"""
A language model feature extractor using kenlm.

:Authors: - Wilker Aziz
"""

from grasp.cfg.symbol cimport Terminal
from grasp.scoring.extractor cimport Stateless, Stateful


DEFAULT_BOS_STRING = '<s>'
DEFAULT_EOS_STRING = '</s>'


cdef class StatelessLM(Stateless):

    cdef:
        int _order
        Terminal _bos
        Terminal _eos
        str _path
        object _model
        tuple _features
        object _initial

    cdef _load_model(self)


cdef class KenLM(Stateful):

    cdef:
        int _order
        Terminal _bos
        Terminal _eos
        str _path
        object _model
        tuple _features
        object _initial

    cdef _load_model(self)


cdef class ConstantLM(Stateful):

    cdef float _constant
