///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
//
///////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fst/fstlib.h>

using namespace std;
using namespace fst;

typedef unsigned long FstIndex;

#define MAX_STRING 200



/****************************************************************************
                                 FUNCTIONS
*****************************************************************************/


/**
 * Compute the conditional probability of a word given a state.
 * This computation handles backoffs in the FST.
 */
float computeFstWordProb(VectorFst<LogArc> &fst, int word, FstIndex &state) {
	Matcher< VectorFst<LogArc> > matcher(fst, MATCH_INPUT);
	matcher.SetState(state);
	LogWeight prob = LogWeight::One();
	while (!matcher.Find(word)) {
		ArcIterator< VectorFst<LogArc> > it(matcher.GetFst(), state);
		
		prob = Times(prob, it.Value().weight); //apply backoff weight
		printf("\t[ BO=%.3f ]\n", exp(-it.Value().weight.Value()));
		state = it.Value().nextstate;
		matcher.SetState(state);
		printf("[%li]", state);
	}
	state = matcher.Value().nextstate;
	prob = Times(prob, matcher.Value().weight.Value());

//	printf("\t\t%i (%li/w%i) --> %i = %f\n", matcher.Value().ilabel , state, word, matcher.Value().nextstate, matcher.Value().weight.Value());
	return prob.Value();
}










/**
 * Compute the conditional probability of a word given a state.
 * This computation handles backoffs in the FST.
 */
float computeFstFinalProb(VectorFst<LogArc> &fst, FstIndex state) {
	LogWeight prob = LogWeight::One();
	while (fst.Final(state) == fst.Final(state).Zero() && state != 0) {
		ArcIterator< VectorFst<LogArc> > it(fst, state);
		prob = Times(prob, it.Value().weight); //apply backoff weight
//		printf("\t\t</s> (%li/eps) --> %i = %f\n", state, it2.Value().nextstate, it2.Value().weight.Value());
		state = it.Value().nextstate;
	}
	prob = Times(prob, fst.Final(state));
//	printf("\t\t</s> (%li/</s>) --> END = %f\n", state, fst.Final(state).Value());
	return prob.Value();
}



/****************************************************************************
                                 MAIN
*****************************************************************************/


void computePerplexity(VectorFst<LogArc> &fst, const char *txt_fn) {
	ifstream in (txt_fn);
	vector<string> words;
	string word;
	string line;
	
	float p = 0.0;;
	float logP=0.0;
	float sum_logP = 0.0;
	int n_wrd = 0;
	int n_utt = 0;
	int n_lines = 0;
	int l;
	const SymbolTable *dict = fst.InputSymbols();
	FstIndex state = fst.Start();
	FstIndex prev_state = 0;


	Matcher< VectorFst<LogArc> > matcher(fst, MATCH_INPUT);
	matcher.SetState(fst.Start());
	cout << dict->NumSymbols() << endl;
	

	if ( !in )
	  return;
	

	//while ( in >> word ) {
	while (getline(in, line)) {
		istringstream strstr(line);
		while (strstr >> word) {
			l = dict->Find(word);
			prev_state = state;
		
// 			//normal word
// 			if (matcher.Find(l)) { 
// 				const LogArc &arc = matcher.Value();
// 				printf("[%i]\t%f\t%s (%i)\n", state, arc.weight.Value(), word.c_str(), l);
// 				logP += log10(arc.weight.Value());
// 				n_wrd++;
// 				n_utt++;
// 			
// 				state = arc.nextstate;
// 			}
			
						//normal word
			if (l > 0) { 
				printf("[%li]", state);
				p = computeFstWordProb(fst, l, state);
				printf("\t%f\t%s (%i)\n", exp(-p), word.c_str(), l);
				logP += p;
				n_wrd++;
				n_utt++;
			
			}
		
			//end of sentence
			else if (word == "</s>") {
				//skip it -> this is handled by the end of line character
			}
		
			//move to the next state
			matcher.SetState(state);
		}
		
		prev_state = state;
		printf("[%li]", state);
		p = computeFstWordProb(fst, 1, state);
		printf("\t%f\t</s> (1)\n", exp(-p));

		logP += p;
		n_wrd+=1; // </s>
	
		sum_logP += logP;
		n_utt+=1; // </s>
	
		n_lines++;
	
		cout << endl;
		printf("------------------------------------------------------------------------\n");
		printf("LogP = %f \tLogP (base 10) = %f \tPPL = %f\n", -logP, -logP/log(10), exp(logP/(float) n_wrd));
		printf("------------------------------------------------------------------------\n");
		//reset
		state = fst.Start();;
		matcher.SetState(fst.Start());
		n_wrd = 0;
		logP = 0;
		//
	}
		printf("========================================================================\n");
	printf("LogP = %f \tLogP (base 10) = %f \tPPL = %f\n", -sum_logP, -sum_logP/log(10), exp(sum_logP/(float) n_utt));
	
	printf("N words = %i \tN sentences = %i\n", n_utt, n_lines);
	cout << endl;
}






void computeInteractivePerplexity(VectorFst<LogArc> &fst) {
	vector<string> words;
	string word;
	string line;
	
	float p = 0.0;;
	float logP=0.0;
	float sum_logP = 0.0;
	int n_wrd = 0;
	int n_utt = 0;
	int n_lines = 0;
	int l;
	const SymbolTable *dict = fst.InputSymbols();
	FstIndex state = 0;
	FstIndex prev_state = 0;


	Matcher< VectorFst<LogArc> > matcher(fst, MATCH_INPUT);
	matcher.SetState(fst.Start());
	cout << dict->NumSymbols() << endl;
	

	//while ( in >> word ) {
	while (getline(cin, line)) {
		istringstream strstr(line);
		while (strstr >> word) {
			prev_state = state;
			l = dict->Find(word);
		
						//normal word
			if (l > 0) { 
				printf("[%li]", state);
				p = computeFstWordProb(fst, l, state);
				printf("\t%f\t%s (%i)\n", exp(-p), word.c_str(), l);
				logP += p;
				n_wrd++;
				n_utt++;
			
			}
		
			//end of sentence
			else if (word == "</s>") {
				//skip it -> this is handled by the end of line character
			}
			
			//OOV
			else {
				printf("[%li]\t", state);
				printf("0.000000\t%s (OOV)\n", word.c_str());
			}
		
			//move to the next state
			matcher.SetState(state);
		}
		
		prev_state = state;
//		p = computeFstFinalProb(fst, state);
				printf("[%li]", state);
				p = computeFstWordProb(fst, 1, state);
				printf("\t%f\t</s> (0)\n", exp(-p));

		logP += p;
		n_wrd+=1; // </s>
	
		sum_logP += logP;
		n_utt+=1; // </s>
	
		n_lines++;
	
		cout << endl;
		printf("------------------------------------------------------------------------\n");
		printf("LogP = %f \tLogP (base 10) = %f \tPPL = %f\n", -logP, -logP/log(10), exp(logP/(float) n_wrd));
		printf("------------------------------------------------------------------------\n");
		//reset
		state = 0;
		matcher.SetState(0);
		n_wrd = 0;
		logP = 0;
		//
	}
	cout << endl;
}


int argPos(char *str, int argc, char **argv)
{
    int a;
    
    for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;
    
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    
    int debug_mode=1;
    int fst_file_set=0;
    int text_data_set = 0;
    int interactive = 0;
    
    char text_file[MAX_STRING];
    char fst_file[MAX_STRING];
    
    FILE *f;
    
    //RNN LM
	VectorFst<LogArc> *fst;
    
    if (argc==1) {
    	//printf("Help\n");

    	printf("Converts a recurrent neural network into a finite state transducer in order to integrate long-span information within the decoding process\n\n");
    	
    	printf("Syntax:\n\twfst-ppl -text <rnn_model> -fst <fst_output>\n\n");

    	return 0;	//***
    }

    
    //set debug mode
    i=argPos((char *)"-debug", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: debug mode not specified!\n");
            return 0;
        }

        debug_mode=atoi(argv[i+1]);

	if (debug_mode>0)
        printf("debug mode: %d\n", debug_mode);
    }

    //set FST file
    i=argPos((char *)"-fst", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: FST file not specified!\n");
            return 0;
        }

        strcpy(fst_file, argv[i+1]);

        if (debug_mode>0)
        printf("FST file: %s\n", fst_file);

        fst_file_set=1;
    }    
    
    //search for text file
    i=argPos((char *)"-text", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: text data file not specified!\n");
            return 0;
        }

        strcpy(text_file, argv[i+1]);

        if (debug_mode>0)
        printf("text file: %s\n", text_file);

        f=fopen(text_file, "rb");
        if (f==NULL) {
            printf("ERROR: text data file not found!\n");
            return 0;
        }
        else {
        	fclose(f);
        }

        text_data_set=1;
    }
    
   //search for text file
    i=argPos((char *)"-interactive", argc, argv);
    if (i>0) {

        if (debug_mode>0)
        printf("iteractive mode: on\n", text_file);
        interactive=1;
    }
    
    
    if (!fst_file_set) {
    	printf("ERROR: WFST is missing!\n");
    	return 0;
    }
    if (!text_data_set && !interactive) {
    	printf("ERROR: you have to define either a text file or to switch on the interactive mode!\n");
    	return 0;
    }
    
	fst = VectorFst<LogArc>::Read(string(fst_file));
	if (interactive) {
		computeInteractivePerplexity(*fst);	
	}
	else {
	computePerplexity(*fst, text_file);
	}
    
    return 0;
}
