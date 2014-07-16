///////////////////////////////////////////////////////////////////////
//
// Specialization of FstBuilder using a different backoff strategy
// based on the search for a flatest distribution for backoff nodes
// instead of the min KL divergence for the classical builder
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////


#include "neuron_fstbuilder.h"

typedef std::pair<real, std::pair<int,int> > dist_dim_val_triple;

class FlatBOFstBuilder : public NeuronFstBuilder {
	protected:
	
	NeuronFstHistory getBackoff(CRnnLM &rnnlm, const NeuronFstHistory &fsth, set<NeuronFstHistory> &set_min_bo, vector<real> &cur_cond, vector<int> &words);
	
	public:
	
	FlatBOFstBuilder(NeuronDiscretizer *d) : NeuronFstBuilder(d) {}
	FlatBOFstBuilder(NeuronDiscretizer *d, real t, int bol) : NeuronFstBuilder(d,t,bol) {}
	void convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst);
};



