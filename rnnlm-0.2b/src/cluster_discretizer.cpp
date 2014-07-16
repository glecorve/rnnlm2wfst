/************************************************************************
 * Mapping a discretized state in the RNN as well as to move back
 * continuous values.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ***********************************************************************/

#include "cluster_discretizer.h"
#include "cluster_fsthistory.h"
#include <iostream>
#include <sstream>

using namespace std;

#define mylog(x) -log(x)

ClusterDiscretizer::ClusterDiscretizer(int dims, int cl
// , int nw
 ) {
	means = new real*[cl];
	prior = new real[cl];
//	word_prior = new real*[cl];
	n_dims = dims;
	n_clusters = cl;
//	n_words = nw;
	for (int i=0; i < cl; i++) {
		means[i] = new real[dims];
//		word_prior[i] = new real[n_words];
	}
}

ClusterDiscretizer::ClusterDiscretizer(int dims, int cl,
// int nw,
 string fn) {
	means = new real*[cl];
	prior = new real[cl];
//	word_prior = new real*[cl];
	n_dims = dims;
	n_clusters = cl;
//	n_words = nw;
	for (int i=0; i < cl; i++) {
		means[i] = new real[dims];
//		word_prior[i] = new real[n_words];
	}
	load(fn);
}

ClusterDiscretizer::ClusterDiscretizer(const ClusterDiscretizer &dzer) {
	n_dims = dzer.n_dims;
	n_clusters = dzer.n_clusters;
//	n_words = dzer.n_words;
	means = new real*[n_clusters];
	prior = new real[n_clusters];
//	word_prior = new real*[n_clusters];
	
	//for each cluster
	for (int i=0; i < n_clusters; i++) {
		//copy cluster prior
		prior[i] = dzer.prior[i];
		//copy mean
		means[i] = new real[n_dims];
		for (int j=0; j < dzer.n_dims; j++) {
			means[i][j] = dzer.means[i][j];
		}
//		//copy word priors
//		word_prior[i] = new real[n_words];
//		for (int j=0; j < dzer.n_words; j++) {
//			word_prior[i][j] = dzer.word_prior[i][j];
//		}
	}
}



real ClusterDiscretizer::distanceL2(const real * const u, const struct neuron * const v) const {
	real dist = 0.0;
	for (int i = 0; i < n_dims; i++) {

		dist += pow(u[i]-v[i].ac, 2);
	}
	return sqrt(dist);
}



////////////////////////////////////
// Implement virtual pure methods //
////////////////////////////////////


void ClusterDiscretizer::discretize(FstHistory* const fsth, const struct neuron * const layer) const {
	ClusterFstHistory *p = dynamic_cast<ClusterFstHistory *>(fsth);
	if (p != NULL) {
		int min_cl = 0;
		real min_dist = 1e100;
		real dist = 0.0;
		for (int i = 0; i < getNumClusters(); i++) {
			dist = distanceL2(means[i],layer);
//			printf("DIST(%i) = %f\n", i, (float) dist);
			if (dist < min_dist) {
				min_dist = dist;
				min_cl = i;
			}
		}
//		printf("FINAL DIST = DIST(%i) = %f\n", min_cl, (float) min_dist);
		p->setDiscretized(min_cl);
	}
}


	
void ClusterDiscretizer::undiscretize(struct neuron * const layer, const FstHistory * const fsth) const {
	const ClusterFstHistory *p = dynamic_cast<const ClusterFstHistory *>(fsth);
	if (p != NULL) {
		for (int i = 0; i < getNumDims(); i++) {
			layer[i].ac = means[p->getDiscretized()][i];
		}
	}
}


bool ClusterDiscretizer::load(fstream &in) {
	string word;
	string line;
	if ( !in )
	  return false;
	  
	int cl = 0;
	while (getline(in, line)) {
//		cout << line << endl;
		istringstream strstr(line);
		int i = 0;
		real v = 0.0;
		strstr >> word;
		//if comment skip
//		if (word == "--") { continue; }
		if (word[0] == '-' && word[1] == '-') { return (cl >= n_clusters); }
		if (word[0] == '#') { continue; }		
		//read mean
		prior[cl] = mylog(atof(word.c_str()));
//		printf("P_prior(c%i) = %f\n", cl, atof(word.c_str()));
		while (strstr >> word) {
			if (word[0] == '#') { break; }
			v = atof(word.c_str());
			means[cl][i] = v;
//			printf("means[%i][%i] = %f\n", cl, i, v);
			i++;
		}
		//read word priors
//		for (int j=0; j<n_words && getline(in,line); j++) {
//			strstr.str (line);
//			v = atof(line.c_str());
//			word_prior[cl][j] = v;
//			printf("P(w%i|c%i) = %f\n", j, cl, v);
//		}
		
		cl++; //move to next cluster mean
//		if (cl >= n_clusters) { return true; }
	}
// // 	//copy the last line until filling all the dims
// // 	for (cl = cl; cl < n_clusters; cl++) {
// // 		prior[cl] = prior[cl-1];
// // 		for (int i=0; i < n_dims; i++) {
// // 			means[cl][i] = means[cl-1][i];
// // 		}
// // 	}

	return (cl >= n_clusters);
}

bool ClusterDiscretizer::load(string fn) {
	fstream in (fn.c_str());
	bool res = load(in);
	in.close();
	return res;
}



