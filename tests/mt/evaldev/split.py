import sys

handlers = {}

for line in sys.stdin:
    refs = line.split(' ||| ')
    for i, ref in enumerate(refs):
        if i not in handlers:
            handlers[i] = open('refs{0}'.format(i), 'w')
        print(ref, file=handlers[i])

for i, h in handlers.items():
    h.close()
