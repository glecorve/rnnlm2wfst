/************************************************************************
 * Mapping a discretized state in the RNN as well as to move back
 * continuous values.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ***********************************************************************/

#include "neuron_discretizer.h"
#include "neuron_fsthistory.h"
#include <iostream>
#include <sstream>

using namespace std;

NeuronDiscretizer::NeuronDiscretizer(int dims, int bins_per_dim) {
	values = new real*[dims];
	bounds = new real*[dims];
	n_dims = dims;
	n_bins = bins_per_dim;
	for (int i=0; i < dims; i++) {
		values[i] = new real[bins_per_dim];
		bounds[i] = new real[bins_per_dim-1];
	}
}

NeuronDiscretizer::NeuronDiscretizer(int dims, int bins_per_dim, string fn) {
	values = new real*[dims];
	bounds = new real*[dims];
	n_dims = dims;
	n_bins = bins_per_dim;
	for (int i=0; i < dims; i++) {
		values[i] = new real[bins_per_dim];
		bounds[i] = new real[bins_per_dim-1];
	}
	load(fn);
}

NeuronDiscretizer::NeuronDiscretizer(const NeuronDiscretizer &dzer) {
	n_dims = dzer.n_dims;
	n_bins = dzer.n_bins;
	values = new real*[n_dims];
	bounds = new real*[n_dims];
	for (int i=0; i < dzer.n_dims; i++) {
		values[i] = new real[n_bins];
		bounds[i] = new real[n_bins-1];
		for (int j=0; i < dzer.n_bins; j++) {
			values[i][j] = dzer.values[i][j];
		}
		for (int j=0; i < dzer.n_bins-1; j++) {
			bounds[i][j] = dzer.bounds[i][j];
		}
	}
}

// NeuronDiscretizer::~Discretizer() {
// 	delete[] values;
// 	delete[] bounds;
// }


// Implement virtual pure methods

void NeuronDiscretizer::discretize(FstHistory* const fsth, const struct neuron* layer) const {
	NeuronFstHistory *p = dynamic_cast<NeuronFstHistory *>(fsth);
	if (p != NULL) {
		for (int i = 0; i < p->getNumDims(); i++) {
			int bin = 0;
			for (bin = 0; bin < n_bins-1; bin++) {
				if (layer[i].ac < bounds[i][bin]) {
					break;
				}
			}
			p->setDim(i,bin);
		}
	}
}



	
void NeuronDiscretizer::undiscretize(struct neuron * const layer, const FstHistory *  const fsth) const {
	const NeuronFstHistory *p = dynamic_cast<const NeuronFstHistory *>(fsth);
	if (p != NULL) {
		for (int i = 0; i < p->getNumDims(); i++) {
			layer[i].ac = values[i][p->getDim(i)];
		}
	}
}

// 
// int NeuronDiscretizer::discretize(int dim, float v) const {
// 	return 0;
// }
// 
// 
// real NeuronDiscretizer::undiscretize(int dim, int bin_v) const {
// 	return values[dim][bin_v];
// }

bool NeuronDiscretizer::load(string fn) {
	fstream in (fn.c_str());
	string word;
	string line;

	if ( !in )
	  return false;

	int dim = 0;
	while (getline(in, line)) {
//		cout << line << endl;
		istringstream strstr(line);
		int i = 0;
		real v = 0.0;
		real b = 0.0;
		int step = 0;
		while (strstr >> word) {
			if (step == 0 && word[0] == '#') { break; }
			if (step == 0) { dim = atoi(word.c_str()); step = 1; }
			else if (step == 1) { v = atof(word.c_str()); step = 2; }
			else if (step == 2) { b = atof(word.c_str()); step = 1;
				values[dim][i] = v;
				bounds[dim][i] = b;
				i++;
			}
		}
		//last value
		values[dim][i] = v;
	}
	//copy the last line until filling all the dims
	for (dim = dim+1; dim < n_dims; dim++) {
		for (int i=0; i < n_bins; i++) {
			values[dim][i] = values[dim-1][i];
		}
		for (int i=0; i < n_bins-1; i++) {
			bounds[dim][i] = bounds[dim-1][i];
		}
	}

	return true;
}



