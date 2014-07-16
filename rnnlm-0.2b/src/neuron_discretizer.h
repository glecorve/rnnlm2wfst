/************************************************************************
 * Mapping a discretized state in the RNN as well as to move back
 * continuous values.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ***********************************************************************/

#ifndef _NEURON_DISCRETIZER_H_
#define _NEURON_DISCRETIZER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "abstract_discretizer.h"
 
using namespace std;
 
class NeuronDiscretizer : public Discretizer {

	protected:
	real **values;
	real **bounds;
	int n_bins;
	
	public:
	
	NeuronDiscretizer(int dims, int bins_per_dim);
	NeuronDiscretizer(int dims, int bins_per_dim, string fn);
	NeuronDiscretizer(const NeuronDiscretizer &dzer);
	
	int getNumBins() { return n_bins; }
	void discretize(FstHistory* const fsth, const struct neuron * const layer) const;
	void undiscretize(struct neuron * const layer, const FstHistory * const fsth) const;
	bool load(string fn);
	
};
 
#endif


 
