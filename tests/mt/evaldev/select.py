import numpy as np

def save_filtered(ipath, opath, selection):
    with open(ipath, 'r') as fi:
        complete = fi.readlines()
    with open(opath, 'w') as fo:
        for y in Y:
            fo.write(complete[y])

N = 1006
M = 200
src = 'training/lm2/tuning/mira_1/eval.dev.input.20150216-082502/test.input'
refs = 'training/lm2/tuning/mira_1/eval.dev.input.20150216-082502/test.refs'
dirs = [('training/lm2/tuning/mira_1/eval.dev.input.20150216-082502', 'lm2'),
        ('training/lm3/tuning/mira_1/eval.dev.input.20150216-085002', 'lm3'),
        ('training/lm4/tuning/mira_1/eval.dev.input.20150218-074912', 'lm4'),
        ('training/lm5/tuning/mira_1/eval.dev.input.20150216-144744', 'lm5')]

X = np.arange(N)
Y = np.random.choice(N, M, replace=False)
Y.sort()

with open('choice', 'w') as fo:
    for y in Y:
        print >> fo, y

prefix = 'dev{0}'.format(M)
save_filtered(src, '{0}.input'.format(prefix), Y)
save_filtered(refs, '{0}.refs'.format(prefix), Y)

for d, o in dirs:
    save_filtered('{0}/test.trans'.format(d), '{0}.{1}.trans'.format(prefix, o), Y)


