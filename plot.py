import pickle
import pylab as pl
import numpy as np
import sys

results = None
with open(sys.argv[1], 'r') as f:
    results = pickle.load(f)

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results


training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12, 8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())

#box = pl.get_position()
#pl.set_position([box.x0, box.y0, box.width * 0.8, box.height])

pl.legend(bbox_to_anchor=(1.2, 1.00)) #loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.show()
