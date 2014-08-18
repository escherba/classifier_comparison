#!/bin/bash

python plot_xy.py \
    --x _ratio \
    --xlabel "topic1 / topic2 ratio threshold" \
    --y homogeneity \
    --ylabel "homogeneity score" \
    --xlim 1 9 \
    --legend_loc "lower right" \
    --title "Spam Enrichment via Factorization of TF-IDF Representation" \
    --files $*
