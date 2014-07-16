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
#include "cluster_fsthistory.h"
 
bool ClusterFstHistory::lower(const FstHistory *fsth) const {
	const ClusterFstHistory *p = dynamic_cast<const ClusterFstHistory *>(fsth);
	if (getLastWord() != fsth->getLastWord()) {
		return (p != NULL) && getLastWord() < p->getLastWord();
	}
	else {
		return (p != NULL) && getDiscretized() < p->getDiscretized();
	}
}

bool ClusterFstHistory::sameDiscretization(const FstHistory *fsth) const {
	const ClusterFstHistory *p = dynamic_cast<const ClusterFstHistory *>(fsth);
	return (p != NULL) &&  getDiscretized() == p->getDiscretized();
}



//Display

string ClusterFstHistory::toString() const {
	ostringstream str;
	str << getLastWord() << " | " << getDiscretized();
	return str.str();
}

