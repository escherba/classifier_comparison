#!/bin/sh
domino run grid_search.py --scoring f1 --data_dir corpora/livefyre --shard 0 --n_shards 2 --output_basename Latest
domino run grid_search.py --scoring f1 --data_dir corpora/livefyre --shard 1 --n_shards 2 --output_basename Latest
