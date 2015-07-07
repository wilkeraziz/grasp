"""
:Authors: - Wilker Aziz
"""
from collections import defaultdict, deque
from .carry import Carry
from .hg import Hypergraph


def make_hg_from_cfg(cfg):
    sym2node = defaultdict(None)
    hg = Hypergraph()
    for sym in cfg.itersymbols():
        n = sym2node.get(sym, None)
        if n is None:
            n = hg.add_node(Carry(symbol=sym))
            sym2node[sym] = n

    for rule in cfg:
        head = sym2node[rule.lhs]
        tail = [sym2node[s] for s in rule.rhs]
        hg.add_edge(head, tail, 0.0, rule)

    return hg


def make_ihg_from_scfg(scfg):
    sym2node = defaultdict(None)
    hg = Hypergraph()
    for sym in scfg.itersigma():
        n = sym2node.get(sym, None)
        if n is None:
            n = hg.add_node(Carry(symbol=sym))
            sym2node[sym] = n

    edges2rule = deque()
    for srule in scfg:
        head = sym2node[srule.lhs]
        tail = [sym2node[s] for s in srule.irhs]
        hg.add_edge(head, tail, 0.0, srule)
        edges2rule.append(srule)

    return hg