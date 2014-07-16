#CC = x86_64-linux-g++-4.5
#CFLAGS = -lm -O2 -funroll-loops -fprefetch-loop-arrays

CC = g++
# CFLAGS = -Wl,--no-as-needed -lm -O2 -Wall -funroll-loops -ffast-math
CFLAGS = -Wl,--no-as-needed -lm -g -O2 -Wall -funroll-loops -ffast-math
BIN=../bin
SRC=src
OPENFST:=../../openfst-1.3.2/
ifeq ($(USE_BLAS),1)
BLAS_LIBS = -L/usr/lib -lblas -latlas
OPT_DEF = -D USE_BLAS
endif


all: rnnlmlib.o rnnlm rnn2fst wfst-ppl compute-mapping trace-hidden-layer

# EXEC


rnnlm : rnnlm.o abstract_discretizer.o abstract_fsthistory.o cluster_discretizer.o cluster_fsthistory.o hierarchical_cluster_discretizer.o hierarchical_cluster_fsthistory.o rnnlmlib.o
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) $^ -o $(BIN)/$@

compute-mapping : compute-mapping.o rnnlmlib.o abstract_discretizer.o abstract_fsthistory.o cluster_discretizer.o cluster_fsthistory.o hierarchical_cluster_discretizer.o hierarchical_cluster_fsthistory.o rnnlmlib.o
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) $^ -o $(BIN)/$@
	
trace-hidden-layer : trace-hidden-layer.o rnnlmlib.o abstract_discretizer.o abstract_fsthistory.o cluster_discretizer.o cluster_fsthistory.o hierarchical_cluster_discretizer.o hierarchical_cluster_fsthistory.o rnnlmlib.o
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) $^ -o $(BIN)/$@

rnn2fst : rnn2fst.cpp rnnlmlib.o abstract_discretizer.o abstract_fsthistory.o abstract_fstbuilder.o neuron_fsthistory.o neuron_discretizer.o neuron_fstbuilder.o flat_bo_fstbuilder.o cluster_discretizer.o cluster_fsthistory.o cluster_fstbuilder.o hierarchical_cluster_discretizer.o hierarchical_cluster_fsthistory.o hierarchical_cluster_fstbuilder.o
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) -I $(OPENFST)/include/ -L$(OPENFST)/lib/ -ldl $(OPENFST)/lib/libfst.so $^ -o $(BIN)/$@

wfst-ppl : wfst-ppl.cpp abstract_discretizer.o abstract_fsthistory.o cluster_discretizer.o cluster_fsthistory.o hierarchical_cluster_discretizer.o hierarchical_cluster_fsthistory.o rnnlmlib.o
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) -I $(OPENFST)/include/ -L$(OPENFST)/lib/ -ldl $(OPENFST)/lib/libfst.so $^ -o $(BIN)/$@


# OBJ


abstract_fstbuilder.o : abstract_fstbuilder.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF)  -I $(OPENFST)/include/ -o $@ -c $^

neuron_fstbuilder.o : neuron_fstbuilder.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF)  -I $(OPENFST)/include/ -o $@ -c $^

flat_bo_fstbuilder.o : flat_bo_fstbuilder.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF)  -I $(OPENFST)/include/ -o $@ -c $^
	
cluster_fstbuilder.o : cluster_fstbuilder.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF)  -I $(OPENFST)/include/ -o $@ -c $^
	
hierarchical_cluster_fstbuilder.o: hierarchical_cluster_fstbuilder.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF)  -I $(OPENFST)/include/ -o $@ -c $^

%.o : %.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) -o $@ -c $<
	

clean:
	rm -rf *.o

