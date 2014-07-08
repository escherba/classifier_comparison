#!/usr/bin/python

from argparse import ArgumentParser
import json

op = ArgumentParser()
op.add_argument('--basename', type=str, default='Untitled',
                help='base name of the files containing shard results')
op.add_argument('--n_shards', type=int, default=1,
                help='number of shards whose results we are combining')
args = op.parse_args()

jj = []
for i in range(args.n_shards):
    f = '%s.%d.txt' % (args.basename, i)
    s = open(f).read()
    j = json.loads(s)
    jj.append(j)

best_score = None
best_i = None
for j in jj:
    score = j[u'best_score']
    if best_score is None or best_score < score:
        best_score = score
        best_i = i

j = jj[best_i]
print 'Best score:', j[u'best_score']
print 'Best parameters:'
for k, v in j[u'best_parameters']:
    print '\t%s: %s' % (k, v)
