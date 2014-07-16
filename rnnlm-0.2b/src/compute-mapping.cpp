///////////////////////////////////////////////////////////////////////
//
// Read a file along with a RNN LM.
// Average values for each dimension of the RNN hidden layer are computed
// and stored into a mapping files.
//
// Gwénolé Lecorvé
//
///////////////////////////////////////////////////////////////////////

#include "rnnlmlib.h"
#include <string>
#include <vector>

int debug_mode = 0;

using namespace std;

void traceHiddenLayer(string fn, CRnnLM &rnnlm, vector<vector<real> > &trace) {
    int a, b, i, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob;
    char str[200];
    real prob_other, log_other, log_combine, f;
    int overwrite;
    real logp;
    struct neuron* in = rnnlm.getInputLayer();    
    struct neuron* hid = rnnlm.getHiddenLayer();
    struct neuron* out = rnnlm.getOutputLayer();
    
    rnnlm.restoreNet();
    
    fi=fopen(fn.c_str(), "rb");

    last_word=0;					//last word = end of sentence
    logp=0;
    log_other=0;
    log_combine=0;
    prob_other=0;
    rnnlm.copyHiddenLayerToInput();
	vector<real> one_trace;
	for (int j = 0; j < rnnlm.getHiddenLayerSize(); j++) {
		one_trace.push_back(hid[j].ac);
	}
	trace.push_back(one_trace);
    
    while (1) {
        
        
        word=rnnlm.readWordIndex(fi);		//read next word
        rnnlm.computeNet(last_word, word);		//compute probability distribution
        
        
        //trace
		vector<real> one_trace;
		for (int j = 0; j < rnnlm.getHiddenLayerSize(); j++) {
			one_trace.push_back(hid[j].ac);
		}
		trace.push_back(one_trace);
		
        if (feof(fi)) break;		//end of file: report LOGP, PPL

        rnnlm.copyHiddenLayerToInput();
        
        if (last_word!=-1) in[last_word].ac=0;  //delete previous activation
        last_word=word;
    }
    fclose(fi);

}






vector<real> computeMeans(vector< vector<real> > &trace, real min, real max) {
	vector<real> output;
	vector<int> n;
	
	for(int i=0; i < trace.size(); i++) {
		for (int j=0; j < trace[i].size(); j++) {
			if (i == 0) {
				output.push_back(0.0);
				n.push_back(0);
			}
			if ((trace[i][j] >= min) && (trace[i][j] <= max)) {
				output[j] += trace[i][j];
				n[j]++;
			}
		}
	}
	if (trace.size() > 0) {
		for (int j=0; j < trace[0].size(); j++) {
			if (n[j] > 0) {
				output[j] /= n[j];
			}
		}
	}
	return output;
}



void rec_binarization(vector<real> & val, vector< vector<real> > &trace, real min, real max, int dim, int n_bins) {
	vector<real> means = computeMeans(trace, min, max);
	if (n_bins == 1) {
		val.push_back(means[dim]);
	}
	else {
		vector<real> var;
		rec_binarization(val, trace, min, means[dim], dim, n_bins/2);
		val.push_back(means[dim]);
		rec_binarization(val, trace, means[dim], max, dim, n_bins/2);
	}
}




/****************************************************************************
                                 MAIN
*****************************************************************************/



int argPos(char *str, int argc, char **argv)
{
    int a;
    
    for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;
    
    return -1;
}

int main(int argc, char **argv)
{

    int i;
    
    
    int rnnlm_file_set=0;
    int txt_file_set=0;
    int disc_map_file_set=0;
    int rnnlm_exist=0;
    int n_bins = 2;
    int bo_len=2;
    float threshold=0.01;
    
    
    char rnnlm_file[MAX_STRING];
    char txt_file[MAX_STRING];
    char disc_map_file[MAX_STRING];
    
    FILE *f;
    
    //RNN LM
	CRnnLM rnnlm;

    
    if (argc==1) {
    	//printf("Help\n");

    	printf("Converts a recurrent neural network into a finite state transducer in order to integrate long-span information within the decoding process\n\n");
    	
    	printf("Syntax:\n\tcompute-mapping -bins <N> -rnnlm <rnn_model> -text <text> -discretize <output_mapping>\n\n");

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

    
    
    //set pruning threshold
    i=argPos((char *)"-prune", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: pruning threshold not specified!\n");
            return 0;
        }

        threshold=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Pruning threshold: %f\n", threshold);
    }
    
    //set maximum backoff path length
    i=argPos((char *)"-backoff", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: pruning threshold not specified!\n");
            return 0;
        }

        bo_len=atoi(argv[i+1]);

        if (debug_mode>0)
        printf("Maximum backoff path length: %i\n", bo_len);
    }
        
        
    //set maximum backoff path length
    i=argPos((char *)"-bins", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: Number of bins not specified!\n");
            return 0;
        }

        n_bins=atoi(argv[i+1]);

        if (debug_mode>0)
        printf("Number of bins: %i\n", n_bins);
    }
   
    //search for discretization map file
    i=argPos((char *)"-discretize", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: no discretization map file specified!\n");
            return 0;
        }

        strcpy(disc_map_file, argv[i+1]);

		disc_map_file_set = 1;
		
        if (debug_mode>0)
        printf("discretization map file: %s\n", disc_map_file);

    }
    
    //search for rnnlm file
    i=argPos((char *)"-rnnlm", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: model file not specified!\n");
            return 0;
        }

        strcpy(rnnlm_file, argv[i+1]);


        if (debug_mode>0)
        printf("rnnlm file: %s\n", rnnlm_file);

        f=fopen(rnnlm_file, "rb");
        if (f!=NULL) {
            rnnlm_exist=1;
        }
        rnnlm_file_set=1;
    }


    //set FST file
    i=argPos((char *)"-text", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: Text file not specified!\n");
            return 0;
        }

        strcpy(txt_file, argv[i+1]);

        if (debug_mode>0)
        printf("Text file: %s\n", txt_file);
        txt_file_set=1;
    }    
    

// 	if (disc_map_file_set == 0) {
//         printf("ERROR: no discretization map file specified! Use option -discretize.\n");
//         return 0;
// 	} 
	
	
    //Load RNN LM
    srand(1);
	rnnlm.setRnnLMFile(rnnlm_file);
	rnnlm.setDebugMode(debug_mode);
	rnnlm.restoreNet();
	
	
	
	//Test RNN
	vector< vector<real> > trace;
	vector< real > val;
	traceHiddenLayer(txt_file, rnnlm, trace);
// 	vector<real> means = computeMeans(trace,0.0,1.0);
// 	for (int i=0; i < means.size(); i++) {
// 		rec_binarization(val, trace, 0.0, means[i], i, n_bins/2);
// 		val.push_back(means[i]);
// 		rec_binarization(val, trace, means[i], 1.0, i, n_bins/2);
// 		
// 		printf("%i", i);
// 		for (int j=0; j < 2*n_bins-1; j++) {
// 			printf("\t%f", val[j]);
// 		}
// 		printf("\n");
// 		
// 		val.clear();
// 	}
// 	
	
    
    return 0;
}








