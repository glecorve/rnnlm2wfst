/************************************************************************
 * Model a discretized state in the RNN by discretizing neuron independently.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 *
 *  <------- N dimensions -------->
 *  dim1=2  dim2=1   ...  dimN=0
 * +---+---+---+---+-   -+---+---+
 * | 1   0 | 0   1 | ... | 0   0 |
 * +---+---+---+---+-   -+---+---+
 * <-2bits->
 *    => each dim in [0,2^n_bits-1]
 *
 ************************************************************************/

#include "neuron_fsthistory.h" 

using namespace std;

bool NeuronFstHistory::sameDiscretization(const FstHistory *fsth) const {
	const NeuronFstHistory *p = dynamic_cast<const NeuronFstHistory *>(fsth);
	return (p != NULL) && (this->discretized == p->discretized);
} 
 


//Display

string NeuronFstHistory::toString() const {
	ostringstream str;
	str << getLastWord() << " | ";
	for (int i = 0; i < n_dims; i++) {
		str << getDim(i) << " ";
	}
	return str.str();
}



//Operators

bool NeuronFstHistory::lower(const FstHistory *fsth) const {
	const NeuronFstHistory *p = dynamic_cast<const NeuronFstHistory *>(fsth);
	if (p != NULL) {
		if (getNumDims() != p->getNumDims()) {
			return getNumDims() < p->getNumDims();
		}
		if (getNumBins() !=  p->getNumBins()) {
			return getNumBins() <  p->getNumBins();;
		}
		if (getLastWord() !=  p->getLastWord()) {
			return getLastWord() <  p->getLastWord();;
		}
		for (int i = 0 ; i < getNumDims(); i++) {
			if (getDim(i) !=  p->getDim(i)) {
				return getDim(i) <  p->getDim(i);
			}
		}
	}
	return false;
}
