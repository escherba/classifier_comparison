#!/usr/bin/env python2

"""
Like Unix seq except with super-geometric steps
"""

import sys

x_from, x_step, x_to = map(float, sys.argv[1:4])
curr_x = x_from
xs = []
while curr_x <= x_to:
    xs.append(curr_x)
    x_step **= x_step
    curr_x *= x_step

print ' '.join(['{:.3f}'.format(x) for x in xs])
