.SUFFIXES:
	MAKEFLAGS += -r

.PHONY: pca clean extract_topics env
PYENV = . env/bin/activate;
PYTHON = . env/bin/activate; python
CORPUS_DIR=~/dev/py-nlp/var/corpora/livefyre
CORPUS_DIR2=~/dev/py-nlp/var/corpora/livefyre/dec17
CORPUS_DIR3=~/dev/py-nlp/var/corpora/livefyre/dec29
OUTPUT=out
PLOT_INTERMEDIATE=$(OUTPUT)/fit_metrics
PLOT_INTERMEDIATE2=$(OUTPUT)/fit_metrics_time

# Without options below, Mac builds will attempt to use GCC which fails to
# produce correct compiles with some recent versions of scipy
ifeq ($(shell uname -s), Darwin)
export CC=clang
export CXX="clang++ -stdlib=libstdc++ -I/usr/include/c++/4.2.1"
export FFLAGS=-ff2c
else
# There is bug/weird behavior in virtualenv in Ubuntu that breaks finding the
# include path. This fixes it and let's Python find Python.h while compiling numpy.
export C_INCLUDE_PATH=/usr/include/python2.7
endif

env: requirements.txt
	test -f env/bin/activate || virtualenv --no-site-packages env
	$(PYENV) pip install -r requirements.txt
	$(PYENV) easy_install ipython
	mkdir -p $(OUTPUT)

.PHONY: freq_spam
freq_spam: ./freq_patterns.py utils/lfcorpus.py
	$(PYTHON) freq_patterns.py --category spam --minsup 1500 $(CORPUS_DIR) | sort -k1nr

.PHONY: freq_ham
freq_ham: ./freq_patterns.py utils/lfcorpus.py
	$(PYTHON) freq_patterns.py --category ham --minsup 150 $(CORPUS_DIR) | sort -k1nr

.PHONY: wordclouds
wordclouds: ./Rscripts/create_wordclouds.R
	bash make_wordclouds.sh

browser.plot: $(PLOT_INTERMEDIATE).browser
browser.plot_time: $(PLOT_INTERMEDIATE2).browser

pylab.plot: $(PLOT_INTERMEDIATE).pylab
pylab.plot_time: $(PLOT_INTERMEDIATE2).pylab

%.browser: %.scores chart/chart.scpt chart/chart.html chart/chart.js
	osascript chart/chart.scpt "file://"$(CURDIR)"/chart/chart.html#.."/$<

%.pylab: %.scores.png %.roc.png
	open -a "Preview" $^

pca: pca.py
	$(PYTHON) $< \
		--method SVD \
		--vectorizer tfidf \
		--data_dir $(CORPUS_DIR)

extract_topics: topic_extraction.py
	$(PYTHON) $< \
		--method NMF \
		--n_samples 320000 \
		--n_topics 2 \
		--topic_ratio 3.0 \
		--H_matrix multiply \
		--ground_tag spam \
		--show_topics \
		--input data/2014-01-14.detail.sorted

ap: affinity_propagation.py
	$(PYTHON) $< \
		--data_dir $(CORPUS_DIR) \
		--n_samples 1000 \
		--n_features 100 \
		--categories spam

.PRECIOUS: $(PLOT_INTERMEDIATE).roc.png $(PLOT_INTERMEDIATE2).roc.png
%.roc.png: %.roc
	$(PYTHON) plot_roc.py --input $^ --output $@

.PRECIOUS: $(PLOT_INTERMEDIATE).scores.png $(PLOT_INTERMEDIATE2).scores.png
%.scores.png: plot_scores.py %.scores
	$(PYTHON) $^ $@

.PRECIOUS: $(PLOT_INTERMEDIATE).roc $(PLOT_INTERMEDIATE).scores
$(PLOT_INTERMEDIATE).roc $(PLOT_INTERMEDIATE).scores: %: classify.py utils/lfcorpus.py
	$(PYTHON) $< \
		--data_dir $(CORPUS_DIR) \
		--vectorizer tfidf \
		--top_terms 100 \
		--chi2_select 400 \
		--output_dir $(dir $@) \
		--output_roc $(basename $@).roc \
		--output     $(basename $@).scores


.PRECIOUS: $(PLOT_INTERMEDIATE2).roc $(PLOT_INTERMEDIATE2).scores
$(PLOT_INTERMEDIATE2).roc $(PLOT_INTERMEDIATE2).scores: %: classify.py utils/lfcorpus.py
	$(PYTHON) $< \
		--top_terms 100 \
		--vectorizer tfidf \
		--data_test $(CORPUS_DIR3) \
		--data_train $(CORPUS_DIR2) \
		--output_dir $(dir $@) \
		--output_roc $(basename $@).roc \
		--output     $(basename $@).scores

grid_search: grid_search.py
	$(PYTHON) $< \
		--scoring f1 \
		--data_dir $(CORPUS_DIR)

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml nosetests.xml

clean:
	find . -type f -name "*.pyc" -exec rm -f {} \;
	rm -f $(OUTPUT)/*
