/************************************************************************
 * Generic class to model a discretized state in the RNN.
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ************************************************************************/

#include "abstract_fsthistory.h" 

using namespace std;
 


/**
 * Discretize continuous inputs coming from the recurrent hidden layer
 */
void FstHistory::setFstHistory(CRnnLM & rnnlm, const Discretizer &dzer) {
	struct neuron* layer = rnnlm.getInputLayer();
	
	//browse all word indices
	for (int i=0; i < rnnlm.getVocabSize(); i++) {
		if (layer[i].ac == 1.0) {
			setLastWord(i);	
		}
	}

	//browse all dimensions
	layer = rnnlm.getHiddenLayer();
	dzer.discretize(this,layer);
}



/**
 * Overwrite the current input layer using the current history
 */
void FstHistory::loadAsInput(CRnnLM & rnnlm, const Discretizer &dzer) const {
	struct neuron* in = rnnlm.getInputLayer();
	
	for (int i=0; i < rnnlm.getVocabSize(); i++) {
		in[i].ac = 0.0;
	}
	
	if (getLastWord() != -1) {
		in[getLastWord()].ac = 1.0;
	}

	
	dzer.undiscretize(in+rnnlm.getVocabSize(), this);
	
}



// Dummy implementations of virtual methods (not declared pure virtual
// because this is a mess otherwise)


bool FstHistory::lower(const FstHistory *other) const {
	return (getLastWord() < other->getLastWord());
}


bool FstHistory::sameDiscretization(const FstHistory *fsth) const {
	return true;
}



string FstHistory::toString() const {
	ostringstream str;
	str << "DUMMY H = [ w" << getLastWord() << " ]";
	return str.str();
}



//Operators

bool FstHistory::equals(const FstHistory *other) const {
	return (last_word == other->getLastWord()) && sameDiscretization(other);
}



bool operator==(const FstHistory &one, const FstHistory &other) {
	return one.equals(&other);
}


bool operator!=(const FstHistory &one, const FstHistory &other) {
	return !(one.equals(&other));
}


bool operator< (const FstHistory &fst1, const FstHistory &fst2) {
	return fst1.lower(&fst2);
}

