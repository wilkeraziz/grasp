import sys
from collections import deque


def pop(Q):
    """Pops a char from the top of Q and returns it"""
    if Q:
        return Q.popleft()
    raise ValueError('PLF ERROR: empty queue')


def top(Q):
    """Inspects the top of the queue"""
    if Q:
        return Q[0]
    raise ValueError('PLF ERROR: empty queue')


def eat_blank(Q):
    """Discards a sequence of blank chars at the top of Q"""
    n = len(Q)
    while Q:
        if Q[0] in ' \t\n':
            Q.popleft()
        else:
            break
    return n - len(Q)


def get_escaped_string(Q, quote="'", escape='\\'):
    """Parse a escaped string between single quotes"""
    eat_blank(Q)
    if pop(Q) != quote:
        raise ValueError("PLF ERROR: expected opening '")
    word = ''
    while Q:
        ch = pop(Q)
        if ch == escape: 
            word += pop(Q)  # escaped char
        elif ch == quote:
            return word
        else:
            word += ch
    raise ValueError("PLF ERROR: expected closing '")


def escape_string(string):
    return string.replace('\\', '\\\\').replace("'", "\\'")
    

def parse_plf_tail(string):
    parts = string.split(',')
    if not parts:
        raise ValueError("PLF ERROR: expected at least the destination state")
    fpairs = []
    if len(parts) > 1:
        for part in parts[:-1]:
            pair = part.split('=')
            if len(pair) != 2:
                raise ValueError("PLF ERROR: expected a key=value pair")
            fpairs.append((pair[0].strip(), pair[1].strip()))
    try:
        state = int(parts[-1].strip())
    except:
        raise ValueError("PLF ERROR: state should be integer")
    return fpairs, state


def parse_plf_arc(Q):
    eat_blank(Q)
    if pop(Q) != '(':
        raise ValueError("PLF ERROR: expected '('")
    word = get_escaped_string(Q)
    eat_blank(Q)
    if pop(Q) != ',':
        raise ValueError("PLF ERROR: expected ','")
    string = ''
    while Q:
        ch = pop(Q)
        if ch == ')':
            fpairs, state = parse_plf_tail(string)
            return word, fpairs, state
        else:
            string += ch
    raise ValueError("PLF ERROR: expected ')'")


def parse_plf_block(Q, parse_element):
    eat_blank(Q)
    if pop(Q) != '(':
        raise ValueError("PLF ERROR: expected '('")
    elements = []
    while True:
        # parses an element
        elements.append(parse_element(Q))
        eat_blank(Q)
        ch = pop(Q)
        if ch == ')':  # closing
            return elements
        elif ch == ',':  # there might be more elements
            eat_blank(Q)  
            if top(Q) == ')':  # there aren't more elements
                pop(Q)
                return elements
            continue
        else:
            raise ValueError("PLF ERROR: excepected ',' or ')' got %s" % ch)
    raise ValueError("PLF ERROR: excepected ')'")


def parse_plf_state(Q):
    return parse_plf_block(Q, parse_plf_arc)


def parse_plf(string):
    """
    Parses a Moses-formatted PLF string.

    Return
    ------
    a list of nodes where
        - each node is represented by a list of arcs
        - each arc is a tuple (word, feature pairs, destination)
        - word is a string (possibly in factor notation)
        - feature pairs is a possibly empty list of pairs (str:key, str:value) 
        - destination is an integer representing the destination state
    """
    Q = deque(string)
    return parse_plf_block(Q, parse_plf_state)


def stringfy(plf):
    """
    Call this with a plf (list of states and arcs) to get a string already formatted according to Moses standards.
    """
    str_states = []
    for state in plf:
        str_arcs = []
        for arc in state:
            label, fpairs, destination = arc
            str_arcs.append("('%s', %s, %s)" % (escape_string(label), ', '.join('%s=%s' % (k, v) for k, v in fpairs), destination))
        str_states.append("(%s)" % ', '.join(str_arcs))
    return "(%s)" % ', '.join(str_states)

