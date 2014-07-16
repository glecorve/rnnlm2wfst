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
#include "abstract_discretizer.h"
#include <string>
#include <vector>

int debug_mode = 0;
bool word_id = false;

using namespace std;

void traceHiddenLayer(string fn, CRnnLM &rnnlm) {
	int n=0;
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
	int j;
	printf("%i\t",n);
	if (word_id) {
		printf("%i\t",last_word);
	}
	for (j = 0; j < rnnlm.getHiddenLayerSize()-1; j++) {
		printf("%.6f\t",hid[j].ac);
	}
	printf("%.6f\n",hid[j].ac);
	fprintf(stderr, "%i\n", n);
	n++;
    while (1) {
        
        word=rnnlm.readWordIndex(fi);		//read next word
        rnnlm.computeNet(last_word, word);		//compute probability distribution
        
        //trace
		vector<real> one_trace;
		printf("%i\t",n);
		if (word_id) {
			printf("%i\t",last_word);
		}
		for (j = 0; j < rnnlm.getHiddenLayerSize()-1; j++) {
			printf("%.6f\t",hid[j].ac);
		}
		printf("%.6f\n",hid[j].ac);
		if (n > 367390 && n < 367410) {
		fprintf(stderr, "%i\t%i\n", last_word, n);
		}
		n++;
        if (feof(fi)) break;		//end of file

        rnnlm.copyHiddenLayerToInput();
        
        if (last_word!=-1) in[last_word].ac=0;  //delete previous activation
        last_word=word;
    }
    fclose(fi);

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
    int rnnlm_exist=0;
    
    char rnnlm_file[MAX_STRING];
    char txt_file[MAX_STRING];
    
    FILE *f;
    
    //RNN LM
	CRnnLM rnnlm;
	
    
    if (argc==1) {
    	//printf("Help\n");

    	fprintf(stderr,"Converts a recurrent neural network into a finite state transducer in order to integrate long-span information within the decoding process\n\n");
    	
    	fprintf(stderr,"Syntax:\n\ttrace-hidden-layer [-with-word-id] -rnnlm <rnn_model> -text <text>\n\n");

    	return 0;	//***
    }

    
    //set debug mode
    i=argPos((char *)"-debug", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: debug mode not specified!\n");
            return 0;
        }

        debug_mode=atoi(argv[i+1]);

	if (debug_mode>0)
        fprintf(stderr,"debug mode: %d\n", debug_mode);
    }
    
    
    //set debug mode
    i=argPos((char *)"-with-word-id", argc, argv);
    if (i>0) {
	word_id = true;
	if (word_id)
        fprintf(stderr,"print word history: yes\n");
    }

    
    
    //search for rnnlm file
    i=argPos((char *)"-rnnlm", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: model file not specified!\n");
            return 0;
        }

        strcpy(rnnlm_file, argv[i+1]);


        if (debug_mode>0)
        fprintf(stderr,"rnnlm file: %s\n", rnnlm_file);

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
            fprintf(stderr,"ERROR: Text file not specified!\n");
            return 0;
        }

        strcpy(txt_file, argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Text file: %s\n", txt_file);
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
	traceHiddenLayer(txt_file, rnnlm);

	
    
    return 0;
}








