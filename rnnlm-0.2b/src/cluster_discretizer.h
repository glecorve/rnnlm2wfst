/************************************************************************
 * Associate a RNN state to a cluster ID and, vice-versa, restore a RNN
 * by associating each cluster ID with its mean.
 *
 *
 * Format for defining a cluster in a file
 * <prior1> <dim11> <dim12> ... <dim1N>
 * <word_prior1>
 * <word_prior2>
 * ...
 * <word_priorV>
 * <prior2> <dim21> <dim22> ... <dim2N>
 * ...
 * <priorK> <dimM1> <dimK2> ... <dimKN>
 * ...
 * --
 *
 *
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Oct. 2011
 ***********************************************************************/

#ifndef _CLUSTER_DISCRETIZER_H_
#define _CLUSTER_DISCRETIZER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "abstract_discretizer.h"
 
using namespace std;
 
class ClusterDiscretizer : public Discretizer {

	protected:
	real **means;
	real *prior;
//	real **word_prior;
	
	int n_clusters;
//	int n_words;
	
	real distanceL2(const real * const u, const struct neuron * const v) const;
	
	public:
	
// 	ClusterDiscretizer(int dims, int cl, int nw);
// 	ClusterDiscretizer(int dims, int cl, int nw, string fn);
	ClusterDiscretizer(int dims, int cl);
	ClusterDiscretizer(int dims, int cl, string fn);
	ClusterDiscretizer(const ClusterDiscretizer &dzer);
	
	int getNumClusters() const { return n_clusters; }
	
	void setMean(int cl, int dim, real val) { means[cl][dim] = val; }
	real getMean(int cl, int dim) const { return means[cl][dim]; }
	
	
	void setPrior(int cl, real val) { prior[cl] = val; }
	real getPrior(int cl) const { return prior[cl]; }
	
//	void setWordPrior(int cl, int word, real val) { word_prior[cl][word] = val; }
//	real getWordPrior(int cl, int word) const { return word_prior[cl][word]; }
	
	
	void discretize(FstHistory* const fsth, const struct neuron* layer) const;
	void undiscretize(struct neuron* layer, const FstHistory* fsth) const;
	
	bool load(fstream &in);
	bool load(string fn);
	
};
 
#endif


 
