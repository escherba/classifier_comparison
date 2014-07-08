#!/usr/bin/python

from argparse import ArgumentParser
import os

op = ArgumentParser()
op.add_argument('--n_shards', type=int, default=1,
                help='number of shards whose results we are combining')
op.add_argument('--use_domino', type=int, default=1,
                help="whether to use domino or run locally")
op.add_argument('--output_basename', type=str, default='Latest',
                help="basename of file containing json dict output for combining")
op.add_argument('--scoring', type=str, default='f1',
                help="scoring method")
op.add_argument('--data_dir', type=str, default='corpora/livefyre',
                help="data directory")
args = op.parse_args()

for i in range(args.n_shards):
    # we have to wait for each domino submission to complete before starting the
    # next one.
    prefix = 'domino run ' if args.use_domino else 'python '
    cmd = """
    %sgrid_search.py
        --scoring %s
        --data_dir %s
        --shard %d
        --n_shards %d
        --output_basename %s
    """ % (prefix, args.scoring, args.data_dir, i, args.n_shards, args.output_basename)
    cmd = cmd.replace('\n', '')
    os.system(cmd)
