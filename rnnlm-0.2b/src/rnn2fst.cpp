///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
//
///////////////////////////////////////////////////////////////////////

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "rnnlmlib.h"
#include "abstract_fstbuilder.h"
#include "neuron_fstbuilder.h"
#include "flat_bo_fstbuilder.h"
#include "cluster_fstbuilder.h"
#include "hierarchical_cluster_fstbuilder.h"

using namespace std;
using namespace fst;


#define MAX_STRING 200






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
    
    int debug_mode=0;

    int rnnlm_file_set=0;
    int fst_file_set=0;
    int disc_map_file_set=0;
    int rnnlm_exist=0;
    int n_bins = 2;
    int bo_len=2;
    float threshold=0.01;
    
    bool cluster = false;
    bool h_cluster = false;
    bool neuron_flat = false;
    bool neuron = false;
    
    
    char rnnlm_file[MAX_STRING];
    char fst_file[MAX_STRING];
    char disc_map_file[MAX_STRING];
    
    FILE *f;
    
    //RNN LM
	CRnnLM rnnlm;

    
    if (argc==1) {
    	//printf("Help\n");

    	printf("Converts a recurrent neural network into a finite state transducer in order to integrate long-span information within the decoding process\n\n");
    	
    	printf("Syntax:\n");
		printf("\trnn2fst [-cluster|-hcluster|-neuron|-flat-neuron] -rnnlm <rnn_model> -fst <fst_output>\n");
    	printf("\t        [-prune <prob_threshold>]\n");
    	printf("\t            Threshold to backoff a word transition.\n");
		printf("\t        [-backoff <N>]\n");
    	printf("\t            Maximum length of a backoff path.\n");

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

    //set FST builder type
    i=argPos((char *)"-neuron", argc, argv);
    if (i>0) {
    neuron = true;
	if (debug_mode>0) printf("FST builder: neuron\n");
    }
    
    i=argPos((char *)"-neuron-flat", argc, argv);
    if (i>0) {
    neuron_flat = true;
	if (debug_mode>0) printf("FST builder: neuron flat\n");
    }

    i=argPos((char *)"-cluster", argc, argv);
    if (i>0) {
    cluster = true;
	if (debug_mode>0) printf("FST builder: cluster\n");
    }
    
    i=argPos((char *)"-hcluster", argc, argv);
    if (i>0) {
    h_cluster = true;
	if (debug_mode>0) printf("FST builder: hierarchical cluster\n");
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
    

	if (disc_map_file_set == 0) {
        printf("ERROR: no discretization map file specified! Use option -discretize.\n");
        return 0;
	} 
	
	
	
	// Real stuff now
	
    //Load RNN LM
    srand(1);
	rnnlm.setRnnLMFile(rnnlm_file);
	rnnlm.setDebugMode(debug_mode);
	rnnlm.restoreNet();
	
	//Declare FST builder
	FstBuilder *builder;
	   
	//Load discretizer
	if (neuron || neuron_flat) {
		NeuronDiscretizer *d = new NeuronDiscretizer(rnnlm.getHiddenLayerSize(), n_bins, string(disc_map_file));
		
		if (neuron_flat) {
			builder = (FlatBOFstBuilder *) new FlatBOFstBuilder(d, threshold, bo_len);
		}
		else {
			builder = (NeuronFstBuilder *) new NeuronFstBuilder(d, threshold, bo_len);
		}
	}
	else if (cluster) {
//		ClusterDiscretizer *d = new ClusterDiscretizer(rnnlm.getHiddenLayerSize(), n_bins, rnnlm.getVocabSize(), string(disc_map_file));
		ClusterDiscretizer *d = new ClusterDiscretizer(rnnlm.getHiddenLayerSize(), n_bins, string(disc_map_file));
		builder = (ClusterFstBuilder *) new ClusterFstBuilder(d);
	}
	else if (h_cluster) {
//		HierarchicalClusterDiscretizer *d = new HierarchicalClusterDiscretizer(rnnlm.getHiddenLayerSize(), rnnlm.getVocabSize(), string(disc_map_file));	
		HierarchicalClusterDiscretizer *d = new HierarchicalClusterDiscretizer(rnnlm.getHiddenLayerSize(), string(disc_map_file));	
		builder = (HierarchicalClusterFstBuilder *) new HierarchicalClusterFstBuilder(d, threshold, bo_len);
	}
	
	builder->setDebugMode(debug_mode);
	
	//Create, fill and save FST
	VectorFst<LogArc> fst;
	fst.SetProperties(kILabelSorted, true);
	builder->convertRNN(rnnlm, fst);
	fst.Write(fst_file);
	
	delete builder;
	
    
    return 0;
}
