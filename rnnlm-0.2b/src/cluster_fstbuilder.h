///////////////////////////////////////////////////////////////////////
//
// Specialization of FstBuilder using a different discretization
// Continuous space states are represented by cluster IDs instead of
// discretized values indepently obtained from each neuron activation value
// 
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////

#ifndef _CLUSTER_FSTBUILDER_H_
#define _CLUSTER_FSTBUILDER_H_

#include "cluster_discretizer.h"
#include "abstract_fstbuilder.h"

typedef std::pair<real, std::pair<int,int> > dist_dim_val_triple;

class ClusterFstBuilder : public FstBuilder {
	
	public:
	
	ClusterFstBuilder(ClusterDiscretizer *d) : FstBuilder(d) {}
	void convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst);
};

#endif

