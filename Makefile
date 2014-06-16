.PHONY: plot clean

plot: fit_metrics.pickle
	python plot.py fit_metrics.pickle

fit_metrics.pickle:
	python classify_spam.py --data_dir /Users/escherba/dev/py-nlp/var/corpora/livefyre --output fit_metrics.pickle

clean:
	rm -f *.pyc
