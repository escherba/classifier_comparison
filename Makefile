.PHONY: plot clean

CORPUS_DIR=/Users/escherba/dev/py-nlp/var/corpora/livefyre

plot: plot.py fit_metrics.csv
	python $^

fit_metrics.csv: classify.py
	python $< \
		--data_dir $(CORPUS_DIR) \
		--output $@

grid_search: grid_search.py
	python $< \
		--data_dir $(CORPUS_DIR)

clean:
	rm -f *.pyc
	rm fit_metrics.csv
