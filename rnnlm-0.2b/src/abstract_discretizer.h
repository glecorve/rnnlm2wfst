/************************************************************************
 * Mapping a discretized state in the RNN as well as to move back
 * continuous values.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ***********************************************************************/

#ifndef _DISCRETIZER_H_
#define _DISCRETIZER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "abstract_fsthistory.h"
#include "rnnlmlib.h"
 
using namespace std;
 
class FstHistory;
 
class Discretizer {

	protected:
	int n_dims;
	
	public:
	
	Discretizer();
	Discretizer(const Discretizer &dzer);
	
	virtual int getNumDims() const {
		return n_dims;
	}
	
	virtual void discretize(FstHistory* const fsth, const struct neuron * const layer) const = 0;	
	virtual void undiscretize(struct neuron * const layer, const FstHistory * const fsth) const = 0;
	virtual bool load(string fn) = 0;
	
};
 
#endif


 
