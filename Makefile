.PHONY: plot clean

plot: fit_metrics.pickle plot.py
	python plot.py fit_metrics.pickle

fit_metrics.pickle: classify.py
	python classify.py \
		--data_dir /Users/escherba/dev/py-nlp/var/corpora/livefyre \
		--output fit_metrics.pickle

grid_search:
	python grid_search_text_feature_extraction.py \
		--data_dir /Users/escherba/dev/py-nlp/var/corpora/livefyre

clean:
	rm -f *.pyc
	rm fit_metrics.pickle
