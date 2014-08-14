#!/bin/bash
    

for ratio in `seq 1.0 0.2 6.0`
do
    echo "Running with ratio=${ratio}" >&2
    python topic_extraction.py \
        --method NMF \
        --n_samples 20000 \
        --n_features 300 \
        --n_topics 20 \
        --topic_ratio $ratio \
        --input "data/2014-01-14.detail.sorted"
done | json2csv -k coeff,nmi,u1,u2 > user_curve_NMF.csv
