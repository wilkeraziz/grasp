#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('k', metavar='k', type=int, nargs='?',
  help='number of classes of terminals (terminal-generating non-terminals)')
args = parser.parse_args()

# print("k = %d" % args.k)

print("""[A] ||| [S] [VT] [O] ||| 1.0
[S] ||| [S] [S] ||| 0.2
[S] ||| [S] 'rpi' [S] [VT] ||| 0.2
[S] ||| [ST] ||| 0.6
[O] ||| [O] [O] ||| 0.2
[O] ||| [S] 'rpi' [S] [VT] ||| 0.2
[O] ||| [OT] ||| 0.6""")

p = 1.0 / float(args.k)

for i in range(1, args.k + 1):
  print("[ST] ||| 'si%d' ||| %f" % (i, p))
  print("[OT] ||| 'oi%d' ||| %f" % (i, p))
  print("[VT] ||| 'vi%d' ||| %f" % (i, p))
