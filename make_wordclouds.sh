#!/bin/bash

ls out/*.csv | sed s/.csv// | while read f; do Rscript Rscripts/create_wordclouds.R $f.csv $f.pdf 100; done
