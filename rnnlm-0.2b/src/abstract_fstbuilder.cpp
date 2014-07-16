///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////


// #include <stdio.h>
// #include <stdlib.h>
// #include "rnnlmlib.h"
#include "abstract_fstbuilder.h"

void FstBuilder::dprintf( int min_dbg_lvl, const char* format, ... ) {
	if (debug_mode >= min_dbg_lvl) {
		va_list args;
		va_start( args, format );
		vfprintf( stdout, format, args );
		va_end( args );
    }
}


/**
 * Print a sublist of neurons' activation values
 */
void FstBuilder::printNeurons (struct neuron* neu, int start, int end) {
        printf("Neurons [ ");
        for (int i = start; i < end; i++) {
		printf("%.3f ", neu[i].ac);
        }
        printf("]\n");
}










/**
 * Add a predecessor to a given state id
 */
void FstBuilder::addPred(map< FstIndex, set<FstIndex> > &m, FstIndex k, FstIndex v) {
	map<FstIndex, set<FstIndex> >::iterator it = m.find(k);
	//if new history, then add
	if (it == m.end()) {
		m[k] = set<FstIndex>();
	}
	m[k].insert(v);
	return;
}









/**
 * Try to add a state in the FST for a given history and return the ID
 * of the new state.
 * If the history already exists, the ID of the corresponding state is
 * just returned and the FST is not modified.
 */
bool FstBuilder::addFstState(FstIndex &id, const FstHistory *h, VectorFst<LogArc> &fst) {
	map<const FstHistory*,int>::iterator it = h2state.find(h);
	//if new history, then add
	if (it == h2state.end()) {
		id = (FstIndex) fst.AddState();
		h2state[h] = id;
		return true;
	}
	else {
		delete h;
		id = (FstIndex) it->second;
		return false;
	}
}











/**
 * Compute all conditionals for a given discretize state (FstHistory)
 * and store them in a vector
 */
vector<real> FstBuilder::computeAllConditionals(CRnnLM &rnnlm, const FstHistory & fsth) {
	vector<real> res(rnnlm.getVocabSize());
	struct neuron* output_layer = rnnlm.getOutputLayer();
	int w=0;
	
	fsth.loadAsInput(rnnlm, *dzer);
	
	//store all conditionals
 	rnnlm.computeClassProbs(fsth.getLastWord());
	for (int c = 0; c < rnnlm.getClassSize(); c++) {
	 	rnnlm.computeClassWordProbs(fsth.getLastWord(), rnnlm.getWordFromClass(0, c));
		for (int i = 0; i < rnnlm.getNumWordsInClass(c); i++) {
			w = rnnlm.getWordFromClass(i, c);
			//compute and store P(w|current_state);
			res[w] =
			   mytimes(
			      mylog(output_layer[rnnlm.getVocabSize()+c].ac),
			      mylog(output_layer[w].ac)
			   );			
		}
	}
	
	return res;
}







/**
 * Compute all conditionals for a given discretize state (FstHistory)
 * and store them in a vector
 */
vector<real> FstBuilder::computeSomeConditionals(CRnnLM &rnnlm, const FstHistory & fsth, vector<int> &words) {
	vector<real> res(rnnlm.getVocabSize());
	struct neuron* output_layer = rnnlm.getOutputLayer();
	int w=0;
	
	fsth.loadAsInput(rnnlm, *dzer);
	
	vector<real> mask(rnnlm.getVocabSize(), MY_LOG_ZERO); // 0, 1 mask to disregard some dims
	
	for (int i = 0; i < words.size(); i++) {
		mask[words[i]] = 0.0;
	}
	
	//store all conditionals
 	rnnlm.computeClassProbs(fsth.getLastWord());
	for (int c = 0; c < rnnlm.getClassSize(); c++) {
	 	rnnlm.computeClassWordProbs(fsth.getLastWord(), rnnlm.getWordFromClass(0, c));
		for (int i = 0; i < rnnlm.getNumWordsInClass(c); i++) {
			w = rnnlm.getWordFromClass(i, c);
			//compute and store P(w|current_state);
			res[w] =
			   mask[w] +
			   mytimes(
			      mylog(output_layer[rnnlm.getVocabSize()+c].ac),
			      mylog(output_layer[w].ac)
			   );			
		}
	}
	
	return res;
}









/**
 * Compute all conditionals for a given discretize state (FstHistory)
 * and store them in a vector
 * Entropy is computed at the same time for speed reasons
 */
void FstBuilder::computeEntropyAndConditionals(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory & fsth) {
	computeEntropyAndConditionals(entropy, res, rnnlm, fsth, 0.0);
}


/**
 * Compute all conditionals for a given discretize state (FstHistory)
 * and store them in a vector
 * Entropy is computed at the same time for speed reasons
 */
void FstBuilder::computeEntropyAndConditionals(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory & fsth, real posterior) {
	struct neuron* output_layer = rnnlm.getOutputLayer();
	real p	 = 0.0;
	real p_joint = 0.0;
	int w=0;
	if (posterior > 0.0) { posterior = -posterior; }
	
	entropy = 0.0;
	
	fsth.loadAsInput(rnnlm, *dzer);
	
	//store all conditionals
 	rnnlm.computeClassProbs(fsth.getLastWord());
	for (int c = 0; c < rnnlm.getClassSize(); c++) {
	 	rnnlm.computeClassWordProbs(fsth.getLastWord(), rnnlm.getWordFromClass(0, c));
		for (int i = 0; i < rnnlm.getNumWordsInClass(c); i++) {
			w = rnnlm.getWordFromClass(i, c);
			p = log(output_layer[rnnlm.getVocabSize()+c].ac)
			    +log(output_layer[w].ac);
			p_joint = posterior+p;
			entropy -= exp(p_joint)*p_joint;
			//compute and store P(w|current_state);
			res[w] = -p;	
		}
	}
}




