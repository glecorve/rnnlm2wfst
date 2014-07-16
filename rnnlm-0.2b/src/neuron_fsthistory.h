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

#ifndef _NEURON_FSTHISTORY_H_
#define _NEURON_FSTHISTORY_H_
 
 
#include "abstract_fsthistory.h"
#include <boost/dynamic_bitset.hpp>


//Maximum is 1 dim = 2^8 bins (1 byte)
#define BOUND_A 0.15
#define VALUE_A 0.1
#define VALUE_OTHER 0.6


using namespace std;


class NeuronFstHistory : public FstHistory {

	protected:
	boost::dynamic_bitset<> discretized;
	int n_dims;
	int n_bins;
	int bits_per_dim;
	
	public:

	// Constructor
	NeuronFstHistory(int dims, int bins_per_dim) : FstHistory() {
 		n_dims = dims;
 		n_bins = bins_per_dim;
 		bits_per_dim = (int) (log2(bins_per_dim)+0.5);
 		discretized = boost::dynamic_bitset<>(n_dims*bits_per_dim);
	}
	
	// Copy constructor
	NeuronFstHistory(const NeuronFstHistory &fsth) : FstHistory(fsth) {
 		n_dims = fsth.getNumDims();
 		n_bins = fsth.getNumBins();
 		bits_per_dim = fsth.getDimSize();
 		discretized = boost::dynamic_bitset<>(fsth.discretized);
	}
	
	
	// Getters
	int getNumDims() const { return n_dims; }
	int getNumBins() const { return n_bins; }
	int getDimSize() const { return bits_per_dim; }

	
	// First dim is 0
	int getDim(int i) const {
		int dimi = 0;
		int index = i*bits_per_dim;
		for (int b = index; b < index+bits_per_dim; b++) {
			dimi |= discretized[b];
			if (b+1 < index+bits_per_dim) { dimi = dimi << 1; } //shift left
		}
		return dimi;
	}
	
	

	int distanceL1(const NeuronFstHistory &fsth) const {
		int dist = (getLastWord()==fsth.getLastWord()?0:1);
		for (int i = 0; i < getNumDims(); i++) {
			dist += abs(getDim(i)-fsth.getDim(i));
		}
		return dist;
	}
	
	
	bool sameDiscretization(const FstHistory *fsth) const;
	bool lower(const FstHistory *fsth) const;
	
	//Setters
	// First dim is 0
	void setDim(int i, int v) {
		int index = i*bits_per_dim;
		int bit = 0;
		for (int b = index+bits_per_dim-1; b >= index; b--) {
			bit = v & 1; //get the 1st (right most) bit
			discretized[b] = bit; //store
			v = v >> 1; //shift right
		}
	}
	
	string toString() const;
	
};



#endif
