/************************************************************************
 * Mapping a discretized state in the RNN as well as to move back
 * continuous values.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ***********************************************************************/


#include "hierarchical_cluster_discretizer.h"
#include "hierarchical_cluster_fsthistory.h"
#include <iostream>
#include <sstream>

#define mylog(x) -log(x)

using namespace std;

HierarchicalClusterDiscretizer::HierarchicalClusterDiscretizer(int dims
// , int nw
 ) {
	n_dims = dims;
//	n_words = nw;
}

HierarchicalClusterDiscretizer::HierarchicalClusterDiscretizer(int dims
// , int nw
, string fn) {
	n_dims = dims;
//	n_words = nw;
	load(fn);
}


HierarchicalClusterDiscretizer::HierarchicalClusterDiscretizer(const HierarchicalClusterDiscretizer &dzer) {
	n_dims = dzer.n_dims;
//	n_words = dzer.n_words;
	levels = dzer.levels;
}




////////////////////////////////////
// Implement virtual pure methods //
////////////////////////////////////


void HierarchicalClusterDiscretizer::discretize(FstHistory* const fsth, const struct neuron * const layer) const {
	HierarchicalClusterFstHistory *p = dynamic_cast<HierarchicalClusterFstHistory *>(fsth);
	if (p != NULL) {
		int min_cl = 0;
		real min_dist = 1e100;
		real dist = 0.0;
		p->resetDiscretization();
		ClusterFstHistory reduced_fsth;
		reduced_fsth.setLastWord(fsth->getLastWord());
		for (int i = 0; i < getNumLevels(); i++) {
			levels[i].discretize(&reduced_fsth, layer);
			p->setDiscretized(i,reduced_fsth.getDiscretized());
		}
	}
}


	
void HierarchicalClusterDiscretizer::undiscretize(struct neuron * const layer, const FstHistory * const fsth) const {
	const HierarchicalClusterFstHistory *p = dynamic_cast<const HierarchicalClusterFstHistory *>(fsth);
	if (p != NULL) {
		for (int i = 0; i < getNumDims(); i++) {
			layer[i].ac = levels.at(p->getNumClusters()-1).getMean(p->getFinestDiscretized(),i);
		}
	}
}


// bool HierarchicalClusterDiscretizer::load(string fn) {
// 	fstream in (fn.c_str());
// 	string word;
// 	string line;
// 
// 	if ( !in )
// 	  return false;
// 
// 	int n = 0; //id of a cluster in the current clustering
// 	vector<int> n_cl; //number of cluster in each clustering
// 	while (getline(in, line)) {
// 		if (line == "--") {
// 		printf("%ith level : %i clusters\n", n_cl.size(), n);
// 			n_cl.push_back(n);
// 			n=0;
// 		}
// 		else if (line != "") {
// 		printf("new cluster in %ith level\n", n_cl.size());
// 		n++;
// 		}
// 	}
// 	if (n > 0) {
// 			printf("%ith level : %i clusters\n", n_cl.size(), n);
// 		n_cl.push_back(n);
// 	}
// 	in.close();
// 	
// 	int cl = 0; //id of a clustering
// 	n=0;
// 	in.open(fn.c_str());
// 	levels.push_back(ClusterDiscretizer(n_dims,n_cl[cl]));
// 	while (getline(in, line)) {
// //		cout << line << endl;
// 		if (line == "--") { //create a cl-th clustering
// 			n=0; //cluster id within the new clustering
// 			cl++;
// 			if (cl >= n_cl.size()) { break; }
// 			levels.push_back(ClusterDiscretizer(n_dims,n_cl[cl]));
// 		}
// 		else {
// 			if (n < levels[cl].getNumClusters()) {
// 				istringstream strstr(line);
// 				int i = 0;
// 				real v = 0.0;
// 				strstr >> word;
// 				if (word[0] == '#') { continue; }		
// 				levels[cl].setPrior(n, mylog(atof(word.c_str())));
// 				while (strstr >> word) {
// 					v = atof(word.c_str());
// 					levels[cl].setMean(n,i,v);
// 					printf("%ith level : means[%i][%i] = %f\n", cl, n, i, v);
// 					i++;
// 				}
// 				n++; //move to next cluster mean
// 			}
// 		}
// 		//copy the last line until filling all the dims
// // 		for (n = n; n < levels[cl].getNumClusters(); n++) {
// // 			for (int i=0; i < n_dims; i++) {
// // 				levels[cl].setMean(n,i,levels[cl].getMean(n-1,i));
// // 			}
// // 		}
// 	}
// 
// 
// 	return true;
// }

bool HierarchicalClusterDiscretizer::load(string fn) {
	fstream in (fn.c_str());
	string word;
	string line;

	if ( !in )
	  return false;

	int n = 0; //id of a cluster in the current clustering
	vector<int> n_cl; //number of cluster in each clustering
	while (getline(in, line)) {
		if (line == "--") {
		printf("%ith level : %i clusters\n", n_cl.size(), n);
			n_cl.push_back(n);
			n=0;
		}
		else if ((line != "") && (line.find(" ") != string::npos)) {
		printf("new cluster in %ith level\n", n_cl.size());
		n++;
		}
	}
	if (n > 0) {
			printf("%ith level : %i clusters\n", n_cl.size(), n);
		n_cl.push_back(n);
	}
	in.close();
	
	int cl = 0; //id of a clustering
	n=0;
	in.open(fn.c_str());
	for (int lvl=0; lvl < n_cl.size(); lvl++) {
//		levels.push_back(ClusterDiscretizer(n_dims, n_cl[lvl], n_words));
		levels.push_back(ClusterDiscretizer(n_dims, n_cl[lvl]));
		levels[lvl].load(in);
		
	}
	in.close();
// 	levels.push_back(ClusterDiscretizer(n_dims,n_cl[cl]));
// 	while (getline(in, line)) {
// //		cout << line << endl;
// 		if (line == "--") { //create a cl-th clustering
// 			n=0; //cluster id within the new clustering
// 			cl++;
// 			if (cl >= n_cl.size()) { break; }
// 			levels.push_back(ClusterDiscretizer(n_dims,n_cl[cl]));
// 		}
// 		else {
// 			if (n < levels[cl].getNumClusters()) {
// 				istringstream strstr(line);
// 				int i = 0;
// 				real v = 0.0;
// 				strstr >> word;
// 				if (word[0] == '#') { continue; }		
// 				levels[cl].setPrior(n, mylog(atof(word.c_str())));
// 				while (strstr >> word) {
// 					v = atof(word.c_str());
// 					levels[cl].setMean(n,i,v);
// 					printf("%ith level : means[%i][%i] = %f\n", cl, n, i, v);
// 					i++;
// 				}
// 				n++; //move to next cluster mean
// 			}
// 		}
// 		//copy the last line until filling all the dims
// // 		for (n = n; n < levels[cl].getNumClusters(); n++) {
// // 			for (int i=0; i < n_dims; i++) {
// // 				levels[cl].setMean(n,i,levels[cl].getMean(n-1,i));
// // 			}
// // 		}
// 	}

	return true;
}



