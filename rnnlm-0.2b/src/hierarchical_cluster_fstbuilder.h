///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////



#ifndef _HIERARCHICAL_CLUSTER_FSTBUILDER_H_
#define _HIERARCHICAL_CLUSTER_NEURON_FSTBUILDER_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fst/fstlib.h>
#include "abstract_fstbuilder.h"
#include "hierarchical_cluster_discretizer.h"
#include "hierarchical_cluster_fsthistory.h"
//#include "backoffstrategy.h"

using namespace std;
using namespace fst;


typedef std::pair<real, int> dist_dim_pair;

class HierarchicalClusterFstBuilder : public FstBuilder {
	
	protected:
	
	int num_bins;
	int max_backoff_path;
	real pruning_threshold;

	//Can be overloaded using inheritance
	virtual HierarchicalClusterFstHistory getBackoff(CRnnLM &rnnlm,
	                              const HierarchicalClusterFstHistory &fsth,
	                              set<HierarchicalClusterFstHistory> &min_bo,
	                              vector<real> &cur_cond,
	                              vector<int> &words);
	                              
// 	virtual HierarchicalClusterFstHistory getBackoff(
//                       CRnnLM &rnnlm,
//                       const HierarchicalClusterFstHistory &fsth,
//                       HierarchicalClusterDiscretizer &dzer,
//                       vector<real> &cur_cond,
//                       vector<int> &words);
	                              
	void computeEntropyAndConditionalsSpecial(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory & fsth, real posterior);

	//Computation of backoff weights
	real computeFstWordProb(VectorFst<LogArc> &fst, int word, FstIndex state);
	void computeOneBackoff(VectorFst<LogArc> &fst, FstIndex src, FstIndex bo);
	void computeAllBackoff(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred);

	//Computation of backoff nodes
	real computeDeltaEntropy(real log_p_post, // -log
                     real log_p_cond, // -log
                     real log_p_cond_bo, // -log 
                     real sum_seen, // real
                     real sum_seen_bo); // real
	void estimateMasses(real *mass1, //out
                   real *mass2, //out
                   CRnnLM & rnnlm,
                   real threshold, //in
                   real p_post, //in
                   vector<real> &all_prob, //in
                   vector<real> &all_bo_prob //in
                   );
	void changeTargetForNode(VectorFst<LogArc> &fst, FstIndex old_target, FstIndex new_target);
	void removePred(map< FstIndex,vector<FstIndex> > &pred, FstIndex dest, FstIndex src);
	void removeStates(const VectorFst<LogArc> &old_fst, VectorFst<LogArc> &new_fst, vector<FstIndex> &to_be_deleted);	
	vector<FstIndex> compactBackoffNodes(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred, vector<bool> &non_bo_pred);
	
	real deltaProb(real p_cond, real p_cond_bo);
	
	real computeTotalEntropy(CRnnLM &rnnlm);
	
	public:
	HierarchicalClusterFstBuilder(HierarchicalClusterDiscretizer* d) : FstBuilder(d) {
		max_backoff_path=2;
		pruning_threshold=0.01;
	}
	
	HierarchicalClusterFstBuilder(HierarchicalClusterDiscretizer* d, real t, int bol) : FstBuilder(d) {
		pruning_threshold=t;
		max_backoff_path=bol;
	}

	//Main method
	virtual void convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst);

};


#endif


