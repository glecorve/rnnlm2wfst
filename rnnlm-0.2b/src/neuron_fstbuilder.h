///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////



#ifndef _NEURON_FSTBUILDER_H_
#define _NEURON_FSTBUILDER_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fst/fstlib.h>
#include "abstract_fstbuilder.h"
#include "neuron_discretizer.h"
#include "neuron_fsthistory.h"
//#include "backoffstrategy.h"

using namespace std;
using namespace fst;


typedef std::pair<real, int> dist_dim_pair;

class NeuronFstBuilder : public FstBuilder {
	
	protected:
	
	int num_bins;
	int max_backoff_path;
	real pruning_threshold;

	//Can be overloaded using inheritance
	virtual NeuronFstHistory getBackoff(CRnnLM &rnnlm,
	                              const NeuronFstHistory &fsth,
	                              set<NeuronFstHistory> &min_bo,
	                              vector<real> &cur_cond,
	                              vector<int> &words);

	//Computation of backoff weights
	real computeFstWordProb(VectorFst<LogArc> &fst, int word, FstIndex state);
	void computeOneBackoff(VectorFst<LogArc> &fst, FstIndex src, FstIndex bo);
	void computeAllBackoff(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred);

	//Computation of backoff nodes
	void changeTargetForNode(VectorFst<LogArc> &fst, FstIndex old_target, FstIndex new_target);
	void removePred(map< FstIndex,vector<FstIndex> > &pred, FstIndex dest, FstIndex src);
	void removeStates(const VectorFst<LogArc> &old_fst, VectorFst<LogArc> &new_fst, vector<FstIndex> &to_be_deleted);	
	vector<FstIndex> compactBackoffNodes(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred, vector<bool> &non_bo_pred);
	
	public:
	NeuronFstBuilder(NeuronDiscretizer* d) : FstBuilder(d) {
		num_bins = d->getNumBins();
		max_backoff_path=2;
		pruning_threshold=0.01;
	}
	
	NeuronFstBuilder(NeuronDiscretizer* d, real t, int bol) : FstBuilder(d) {
		num_bins = d->getNumBins();
		pruning_threshold=t;
		max_backoff_path=bol;
	}

	int getNumBins() { return num_bins; }
	
	//Main method
	virtual void convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst);

};


#endif


