rnnlm2wfst
==========

Conversion of recurrent neural network language models to weighted finite state transducers
This directory contains all the code to run the conversion

Gwénolé Lecorvé
Idiap Research Institute, Martigny, Switzerland
2011-2012

How to cite
-----------
	@inproceedings{lecorve2012conversion,
	  title={Conversion of recurrent neural network language models to weighted finite state transducers for automatic speech recognition},
	  author={Lecorv{\'e}, Gw{\'e}nol{\'e} and Motlicek, Petr},
	  booktitle={Thirteenth Annual Conference of the International Speech Communication Association},
	  year={2012}
	}


Prerequisites
-------------

OpenFst: You can use your own version. In that case, edit the makefile in rnnlm-0.2b/src.
BLAS: Better if you have it installed (much faster).

Configuration & compilation
---------------------------

Same as install.sh.

### OpenFst
	cd openfst-1.2.0
	./configure --prefix=`pwd`
	make
	cd ..
	
### K-means
	cd kmeans
	make
	cd ..
	
### RNNLM
	cd rnnlm-0.2b
	# Do not use USE_BLAS=1 if BLAS is not installed
	make USE_BLAS=1
	cd ..

Examples
--------

### RNNLM basic usage
	See examples/example_rrnlm.sh

### Train RNNLM
	bin/rnnlm -train examples/rnn2wfst.train.txt -valid examples/rnn2wfst.dev.txt -rnnlm examples/rnn2wfst.model -hidden 2 -rand-seed 1 -bptt 3 -debug 2 -class 1

### Write logs of continuous states
	bin/trace-hidden-layer -rnnlm examples/rnn2wfst.model -text examples/rnn2wfst.train.txt >  examples/rnn2wfst.train.trace
	
### Generate artificial data (if you think training data is too small or anything else)
	bin/rnnlm -rnnlm examples/rnn2wfst.model -gen 10000 | tail -n +2 > examples/rnn2wfst.generated.txt
	bin/trace-hidden-layer -rnnlm examples/rnn2wfst.model -text examples/rnn2wfst.generated.txt >  examples/rnn2wfst.generated.trace
	
### Build K-means (flat or hiearchical)
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 4 > examples/rnn2wfst.4.kmeans
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 1 8 > examples/rnn2wfst.1+8.kmeans
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 1 2 4 8 > examples/rnn2wfst.1+2+4+8.kmeans

### Cluster-based convertion
	time bin/rnn2fst -rnnlm examples/rnn2wfst.model -fst examples/rnn2wfst.k1+8.p1e-3.fst -discretize examples/rnn2wfst.1+8.kmeans -hcluster -prune 1e-3 -backoff 2
	
	Remark: the value of the backoff option (2) is the depth of the cluster hieararchy.
	
### See the resulting WFST
	fstprint examples/rnn2wfst.k1+8.p1e-3.fst

### Simulate perplexity with discretized RNNLM but without pruning
	bin/rnnlm -rnnlm examples/rnn2wfst.model -test examples/rnn2wfst.test.txt -debug 2 -discretize examples/rnn2wfst.1+8.kmeans | less

### Measure perplexity
	bin/wfst-ppl -fst examples/rnn2wfst.k1+8.p1e-3.fst -text examples/rnn2wfst.test.txt | less
	
### Describe WFST
	../openfst-1.3.2/bin/fstinfo --info_type=long examples/rnn2wfst.k1+8.p1e-3.fst
