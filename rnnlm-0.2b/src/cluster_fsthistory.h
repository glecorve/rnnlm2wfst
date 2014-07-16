/************************************************************************
 * Model a discretized state in the RNN by splicing the continuous space
 * in clusters.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Oct. 2011
 *
 *  N neurons <-----> 1 cluster ID
 *
 ************************************************************************/

#ifndef _CLUSTER_FSTHISTORY_H_
#define _CLUSTER_FSTHISTORY_H_
 
#include "abstract_fsthistory.h"
 
typedef short cluster_id;
 
class ClusterFstHistory : public FstHistory {
	protected:
	cluster_id discretized;
	
	public:
	
	ClusterFstHistory() : FstHistory() {
		discretized = 0;
	}

	ClusterFstHistory(const ClusterFstHistory &fsth) : FstHistory(fsth) {
		discretized = fsth.getDiscretized();
	}
	
	//Getters / Setters
	cluster_id getDiscretized() const { return discretized; }
	void setDiscretized(cluster_id d) { discretized = d; }
	
	// Interface methods
	virtual bool lower(const FstHistory *other) const;
	virtual bool sameDiscretization(const FstHistory *fsth) const;
	string toString() const;
	
};

#endif

