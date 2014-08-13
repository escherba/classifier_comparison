#!/bin/bash
    

for ratio in `seq 1.0 0.2 6.0`
do
    echo "Running with ratio=${ratio}" >&2
    python topic_extraction.py \
        --n_samples 20000 \
        --method NMF \
        --topic_ratio $ratio \
        --n_topics 20 \
        --n_features 300 \
        --ground_tag spam \
        --input "data/2014-01-14.detail.sorted"
done > spam_curve_NMF.json
