/************************************************************************
 * Models a discretized state in the RNN by a an array of sequence IDs
 * for different level of clusters
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ************************************************************************/

#include "hierarchical_cluster_fsthistory.h" 

using namespace std;

bool HierarchicalClusterFstHistory::sameDiscretization(const FstHistory *fsth) const {
	const HierarchicalClusterFstHistory *p = dynamic_cast<const HierarchicalClusterFstHistory *>(fsth);
	return (p != NULL) && (this->discretized == p->discretized);
} 
 


//Display

string HierarchicalClusterFstHistory::toString() const {
	ostringstream str;
	str << getLastWord() << " | ";
	int i;
	if (discretized.size() > 0) {
		for (i = 0; i < (int) discretized.size()-1; i++) {
			str << discretized[i] << "<";
		}
		str << discretized[i];
	}
	else {
		str << "∅";
	}
	return str.str();
}



//Operators

bool HierarchicalClusterFstHistory::lower(const FstHistory *fsth) const {
	const HierarchicalClusterFstHistory *p = dynamic_cast<const HierarchicalClusterFstHistory *>(fsth);
	if (p != NULL) {
		if (getLastWord() !=  p->getLastWord()) {
			return getLastWord() <  p->getLastWord();;
		}
		else if (getNumClusters() != p->getNumClusters()) {
			return getNumClusters() < p->getNumClusters();
		}
		else {
			return discretized < p->discretized;
		}
	}
	return false;
}


