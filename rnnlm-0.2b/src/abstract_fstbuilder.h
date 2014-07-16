///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
// This class is virtual it defines the main function for every inherited class
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////



#ifndef _FSTBUILDER_H_
#define _FSTBUILDER_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fst/fstlib.h>
#include "rnnlmlib.h"
#include "abstract_fsthistory.h"
#include "abstract_discretizer.h"
//#include "backoffstrategy.h"

using namespace std;
using namespace fst;


#define MY_LOG_ZERO 1e100
#define MY_LOG_ONE 0.0
#define mylog(x) -log(x)
#define myexp(x) exp(-x)
#define mytimes(x,y) x+y
#define myplus(x,y) -log(exp(-x)+exp(-y))
#define FstWord(w) w+1
#define INIT_STATE 0
#define FINAL_STATE (unsigned long) 1
#define EPSILON 0
#define WORD_ID(w) w+1


typedef unsigned long FstIndex;
typedef unsigned long StateId;





class FstBuilder {
	
	protected:
	
	int debug_mode;
	
	Discretizer *dzer; //Pointer to allow for dynamic cast over an abstract class
	int max_backoff_path;
	real pruning_threshold;

	map<const FstHistory*,int, FstHistoryCmp> h2state;
	vector<const FstHistory*> state2h;
	
	
	//Basic methods
	void addPred(map< FstIndex, set<FstIndex> > &m, FstIndex k, FstIndex v);
	bool addFstState(FstIndex &id, const FstHistory *h, VectorFst<LogArc> &fst);
	
	//Computation of probs
	vector<real> computeAllConditionals(CRnnLM &rnnlm, const FstHistory &fsth);
	vector<real> computeSomeConditionals(CRnnLM &rnnlm, const FstHistory &fsth, vector<int> &words);
	void computeEntropyAndConditionals(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory &fsth);
	void computeEntropyAndConditionals(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory &fsth, real posterior);
	
	//Debug
	void dprintf( int min_dbg_lvl, const char* format, ... );
	static void printNeurons (struct neuron* neu, int start, int end);
	
	public:
	FstBuilder(Discretizer *d) {
		debug_mode = 0;
		dzer = d;
	}
	
	~FstBuilder() {
		const FstHistory *key;
		map<const FstHistory*,int, FstHistoryCmp>::iterator mit;
		for (mit = h2state.begin(); mit != h2state.end(); ++mit) {
			key = mit->first;
			h2state.erase(mit);
			delete key;
		}
		vector<const FstHistory*>::iterator vit;
		for (vit = state2h.begin(); vit != state2h.end(); ++vit) {
			key = *vit;
			delete key;
		}
		state2h.clear();
	}

	void setDebugMode(int lvl) {
		debug_mode = lvl;
	}
	
	//Static methods
	static real distanceL2(vector<real> u, vector<real> v) {
		real dist = 0.0;

		if (u.size() != v.size()) { return 0.0; }

		for (int i = 0; i < u.size(); i++) {

			dist += pow(exp(-u[i])-exp(-v[i]), 2);
			//printf(" + (%e - %e)^2 = %e = %f\n", exp(-u[i]), exp(-v[i]), pow(exp(-u[i])-exp(-v[i]), 2), dist);
		}

		return sqrt(dist);
	}

	static real distanceKL(vector<real> u, vector<real> v) {
		real dist = 0.0;

		if (u.size() != v.size()) { return 1e100; }

		for (int i = 0; i < u.size(); i++) {
			dist += exp(-u[i])*(v[i]-u[i]); //v - u because of -log
//			printf(" + %e * (%e - %e) = %e = %f\n", exp(-u[i]), -u[i], -v[i], pow(exp(-u[i])-exp(-v[i]), 2), dist);
		}

	//	printf(" = %f\n--\n", dist);

		return dist;
	}
	
	//Main method : to be implemented with inheritage
	virtual void convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) = 0;

};


#endif


