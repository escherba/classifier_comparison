.PHONY: plot clean extract_topics

CORPUS_DIR=/Users/escherba/dev/py-nlp/var/corpora/livefyre

plot: plot.py fit_metrics.csv
	python $^

extract_topics: topic_extraction_with_nmf.py
	python $< \
		--data_dir $(CORPUS_DIR) \
		--n_samples 10000 \
		--n_topics 20 \
		--n_features 4000 \
		--categories spam ham

fit_metrics.csv: classify.py
	python $< \
		--top_terms 100 \
		--data_dir $(CORPUS_DIR) \
		--output $@

grid_search: grid_search.py
	python $< \
		--data_dir $(CORPUS_DIR)

clean:
	rm -f *.pyc
	rm fit_metrics.csv