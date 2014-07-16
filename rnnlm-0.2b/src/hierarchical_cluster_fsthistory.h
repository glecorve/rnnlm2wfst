/************************************************************************
 * Model a discretized state in the RNN by splicing the continuous space
 * in clusters.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Oct. 2011
 *
 *  1 neuron   <----->   ID of cluster in 1st clustering
 *                     | ID of cluster in 2nd clustering
 *                     | ...
 *                     | ID of cluster in Nth clustering
 *
 ************************************************************************/

#ifndef _HIERARCHICAL_CLUSTER_FSTHISTORY_H_
#define _HIERARCHICAL_CLUSTER_FSTHISTORY_H_
 
#include "cluster_fsthistory.h"
 
class HierarchicalClusterFstHistory : public FstHistory {
	protected:
	vector<cluster_id> discretized;
	
	public:

	
	HierarchicalClusterFstHistory() : FstHistory() {
	}
	
	HierarchicalClusterFstHistory(const HierarchicalClusterFstHistory &fsth) : FstHistory(fsth) {
		discretized = fsth.getDiscretized();
	}
	
	//Getters / Setters
	vector<cluster_id> getDiscretized() const { return discretized; }
	cluster_id getFinestDiscretized() const { return discretized.back(); }
	int getNumClusters() const { return (int) discretized.size(); }
	
	void setDiscretized(int lvl, cluster_id d) { 
		if (discretized.size() <= lvl) {
			for (int i=discretized.size(); i <= lvl; i++) {
				discretized.push_back(0);
			}
		}
		discretized[lvl] = d;
	}
	void reduceDiscretization() { discretized.pop_back(); }
	void resetDiscretization() { discretized.clear(); }
	
	// Interface methods
	virtual bool lower(const FstHistory *other) const;
	virtual bool sameDiscretization(const FstHistory *fsth) const;
	string toString() const;
	
};

#endif

