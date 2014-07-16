/************************************************************************
 * Associate a RNN state to a cluster ID and, vice-versa, restore a RNN
 * by associating each cluster ID with its mean.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Oct. 2011
 ***********************************************************************/

#ifndef _HIERARCHICAL_CLUSTER_DISCRETIZER_H_
#define _HIERARCHICAL_CLUSTER_DISCRETIZER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "cluster_discretizer.h"
#include "utils.h"
 
using namespace std;
 
class HierarchicalClusterDiscretizer : public Discretizer {

	protected:
	vector<ClusterDiscretizer> levels;
	int n_words;
	real distanceL2(const real * const u, const struct neuron * const v) const;
	
	public:
	
// 	HierarchicalClusterDiscretizer(int dims, int nw);
// 	HierarchicalClusterDiscretizer(int dims, int nw, string fn);
	HierarchicalClusterDiscretizer(int dims);
	HierarchicalClusterDiscretizer(int dims, string fn);
	HierarchicalClusterDiscretizer(const HierarchicalClusterDiscretizer &dzer);
	
	int getNumLevels() const { return levels.size(); }
	int getLevelSize(int lvl) { return levels[lvl].getNumClusters(); }
	real getPrior(int lvl, int cl) { return levels[lvl].getPrior(cl); }
//	real getWordPrior(int lvl, int cl, int w) { return levels[lvl].getWordPrior(cl,w); }
	void discretize(FstHistory* const fsth, const struct neuron* layer) const;
	void undiscretize(struct neuron* layer, const FstHistory* fsth) const;
	bool load(string fn);
	
};
 
#endif


 
