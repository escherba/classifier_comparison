#!/bin/bash

python topic_extraction.py \
		--method NMF \
		--n_samples 320000 \
		--n_topics $1 \
		--topic_ratio `python seq_sg.py 1.0 1.05 10.0` \
		--H_matrix multiply \
		--ground_attr user \
		--input $2
