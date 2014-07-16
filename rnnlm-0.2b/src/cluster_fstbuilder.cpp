///////////////////////////////////////////////////////////////////////
//
// Specialization of FstBuilder using a different discretization
// Continuous space states are represented by cluster IDs instead of
// discretized values indepently obtained from each neuron activation value
// 
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////

#include "cluster_fstbuilder.h"
#include "cluster_fsthistory.h"


///////////////////////
// CONVERTION METHOD //
///////////////////////



/**
 * Create an FST based on an RNN
 */
void ClusterFstBuilder::convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) {
	queue<ClusterFstHistory> q;
	VectorFst<LogArc> new_fst;
	
	ClusterFstHistory fsth;
	FstIndex id = 0;
	
	ClusterFstHistory new_fsth;
	FstIndex new_id;

	real entropy = 0.0;
	vector<real> all_prob(rnnlm.getVocabSize());
	real p, p_joint;
 	vector<real> posterior(10);
	
	vector<int> to_be_added;
	vector<real> to_be_added_prob;


 	FstIndex n_added = 0;
 	FstIndex n_processed = 0;
 	FstIndex next_n_added = 0;
 	FstIndex next_n_processed = 0;
 	
	int v = rnnlm.getVocabSize();
	int w = 0;


	// Initialize
	rnnlm.copyHiddenLayerToInput();

	// Initial state ( 0 | hidden layer after </s>)
	fsth.setFstHistory(rnnlm, *dzer);
//	printNeurons(rnnlm.getHiddenLayer(),0,2);
	fsth.setLastWord(0);
	q.push(fsth);
	addFstState(id, new ClusterFstHistory(fsth), fst);
	fst.SetStart(INIT_STATE);
 	/*posterior.at(INIT_STATE) = MY_LOG_ONE;*/

	// Final state (don't care about the associated discrete representation)
	fst.AddState();
	fst.SetFinal(FINAL_STATE, LogWeight::One());

	
	
	//foreach state in the queue
	while (!q.empty()) {
		fsth = q.front();
		q.pop();
		id = h2state[&fsth];
		state2h.push_back(new ClusterFstHistory(fsth));
		if (id == FINAL_STATE) { continue; }
		
		
	dprintf(1,"-- STUDY STATE %li = %s\n", id, fsth.toString().c_str());
	

/*		try { posterior.at(id) = MY_LOG_ONE; }
		catch (exception e) {
			posterior.resize((int) (posterior.size()*1.5)+1);
			posterior.at(id) = MY_LOG_ONE;
		}*/
		
		computeEntropyAndConditionals(entropy, all_prob, rnnlm, fsth);
	 	
		//foreach w (ie, foreach word of each class c)
		//test if the edge has to kept or removed
		for (w=0; w < rnnlm.getVocabSize(); w++) {
				p = all_prob[w];
				
				/*p_joint = exp(-posterior[id]-p);*/
				p_joint = exp(-p);
				
				//accept edge if this leads to a minimum
				//relative gain of the entropy

				dprintf(1,"P = %e \tP_joint = %e \tH = %e \tDelta =%e\n",exp(-p), p_joint, entropy);

				next_n_added++;
				to_be_added.push_back(w);
				to_be_added_prob.push_back(p);
				dprintf(1,"\tACCEPT [%li] -- %i (%s) / %f --> ...\n", id, w, rnnlm.getWordString(w), p);
 				
 				//print
				if (next_n_processed % 100000 == 0) {
						fprintf(stderr, "\rH=%.5f / N proc'd=%li / N added=%li (%.5f %%) /%li/%li Nodes (%2.1f %%)", entropy, n_processed, n_added, ((float) n_added/ (float)n_processed)*100.0, id, id+q.size(), 100.0 - (float) (100.0*id/(id+q.size())));
				}
				next_n_processed++;
 				
//			}
		}


		//Set a part of the new FST history
		new_fsth.setFstHistory(rnnlm,*dzer);
//		printNeurons(rnnlm.getHiddenLayer(),0,2);
	
		vector<real>::iterator it_p = to_be_added_prob.begin();
		for (vector<int>::iterator it = to_be_added.begin(); it != to_be_added.end(); ++it) {
			w = *it;
			p = *it_p;

			if (w == 0) {
				fst.AddArc(id, LogArc(FstWord(w),FstWord(w),p,FINAL_STATE));
				dprintf(1,"EDGE [%li] (%s)\n---- %i (%s) / %f -->\n---- [%li] FINAL STATE)\n\n", id, fsth.toString().c_str(), FstWord(w), rnnlm.getWordString(w), p, FINAL_STATE);				
			}
		
			//accept edge
			else {
				new_fsth.setLastWord(w);
	
				//if sw not in the memory
				//then add a new state for sw in the FST and push sw in the queue
				if (addFstState(new_id, new ClusterFstHistory(new_fsth), fst)) {
					q.push(new_fsth);
				}
				else { /* already exists */ }
			
				//add the edge in the FST
				fst.AddArc(id, LogArc(FstWord(w),FstWord(w),p,new_id));
				dprintf(1,"EDGE [%li] (%s)\n---- %i (%s) / %f -->\n---- [%li] (%s)\n\n", id, fsth.toString().c_str(), FstWord(w), rnnlm.getWordString(w), p, new_id, new_fsth.toString().c_str());				

//				posterior.at(new_id) += posterior[id]*p;

			}
			
			++it_p;
		}
		
		n_added = next_n_added;
		n_processed = next_n_processed;
		
		//reset queues
		to_be_added.clear();
		to_be_added_prob.clear();
	}

	cout << endl;
	

	//Fill the table of symbols
	SymbolTable dic("dictionnary");
	dic.AddSymbol("*", 0);
	for (int i=0; i<rnnlm.getVocabSize(); i++) {
		dic.AddSymbol(string(rnnlm.getWordString(i)), i+1);
	}
	fst.SetInputSymbols(&dic);
	fst.SetOutputSymbols(&dic);

	cout << "END" << endl;
	
}



