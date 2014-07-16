/************************************************************************
 * Models a discretized state in the RNN by a an array of sequence IDs
 * for different level of clusters
 *
 * Author: Gwénolé Lecorvé
 * Organisation: IDIAP Research Institue
 * Date: Sept. 2011
 ************************************************************************/

#ifndef _FSTHISTORY_H_
#define _FSTHISTORY_H_
 
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include "utils.h"
#include "rnnlmlib.h"
#include "abstract_discretizer.h"
//#include <iostream>


using namespace std;


class Discretizer;
class CRnnLM;

class FstHistory {

	protected:
	
	int last_word;
	
	public:
	
	// Constructor
	FstHistory() { last_word = -1; }
	FstHistory(const FstHistory& fsth) { last_word = fsth.getLastWord(); }
	
	// Getters
	int getLastWord() const { return last_word; }

	//Setters
	void setLastWord(int w) { last_word = w; }
	
	// Interface methods
	void setFstHistory(CRnnLM & rnnlm, const Discretizer &dzer);
	void loadAsInput(CRnnLM & rnnlm, const Discretizer &dzer) const;
	virtual bool equals(const FstHistory *other) const;
	virtual bool lower(const FstHistory *other) const;
	virtual bool sameDiscretization(const FstHistory *fsth) const;
	virtual string toString() const;
	
	//Operators
	friend bool operator==(const FstHistory &one, const FstHistory &other);
	friend bool operator!=(const FstHistory &one, const FstHistory &other);
	friend bool operator< (const FstHistory &fst1, const FstHistory &fst2);
	
};

struct FstHistoryCmp
{
    bool operator() (const FstHistory* const a, const FstHistory* const b)
    {
        return a->lower(b);  
    }
};

#endif
