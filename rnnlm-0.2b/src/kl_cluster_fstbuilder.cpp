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

#include "kl_cluster_fstbuilder.h"
#include "cluster_fsthistory.h"


///////////////////////
// CONVERTION METHOD //
///////////////////////



/**
 * Return the backoff FST state for a given backed off FST state
 * and eventually update the set of minimal backoff nodes
 */
HierarchicalClusterFstHistory KLClusterFstBuilder::getBackoff(
                      CRnnLM &rnnlm,
                      const ClusterFstHistory &fsth,
                      ClusterDiscretizer &dzer,
                      vector<real> &cur_cond,
                      vector<int> &words)
{

	ClusterFstHistory bo(fsth);
	vector<real> bo_cond(cur_cond.size());
	int k = fsth.getDiscretized();
	float kl_div = 0.0;
	int min_k = k;
	float min_kl_div = 10000.0;
	
	for (int i=0; i<dzer.getNumClusters(); i++) {
		if (i != k && dzer.getPrior(i) > dzer.getPrior(k)) {
			bo.setHistory(i);
			bo_cond = computeSomeConditionals(rnnlm, bo, words);
			kl_dist = distanceKL(cur_cond, bo_cond);
			if (kl_dist < min_kl_div) {
				min_kl_div = kl_div;
				min_k = i;
			}
		}
	}
	if (min_k == k) {
		bo.setLastWord(-1);
	}
	else {
		bo.setHistory(min_k);
	}
	
	return bo;
}


/* ========================================================================================================
                                            CONVERTION METHOD
   ======================================================================================================== */







/**
 * Create an FST based on an RNN
 */
void KLClusterFstBuilder::convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) {
	queue<ClusterFstHistory> q;
	VectorFst<LogArc> new_fst;
	ClusterDiscretizer *typed_dzer = dynamic_cast<ClusterDiscretizer *>(dzer);
	
	ClusterFstHistory fsth;
	FstIndex id = 0;
	
	ClusterFstHistory new_fsth;
	FstIndex new_id;

	ClusterFstHistory min_backoff;
	set<ClusterFstHistory>set_min_backoff;
	
	ClusterFstHistory bo_fsth;
	bool backoff = false;
	vector<FstIndex> deleted;

	real p = 0.0;
	real p_post = 0.0;
	real p_hid = 0.0;
	real p_w_hid = 0.0;
	real bo_post = 0.0;
	real p_joint = 0.0;
	real entropy = 0.0;
	real bo_entropy = 0.0;
	real delta = 0.0;
	real bo_delta = 0.0;
	real mass1 = 1.0;
	real mass2 = 1.0;
	vector<real> all_prob(rnnlm.getVocabSize());
	vector<real> all_bo_prob(rnnlm.getVocabSize());
	map< int , real > entropies;
	
	map< FstIndex,set<FstIndex> > pred;
	vector<bool> non_bo_pred(rnnlm.getVocabSize());
	vector<int> to_be_added;
	vector<int> to_be_removed;
	vector<real> to_be_added_prob;


 	FstIndex n_added = 0;
 	FstIndex n_processed = 0;
 	FstIndex next_n_added = 0;
 	FstIndex next_n_processed = 0;
 	FstIndex n_backoff = 0;
 	FstIndex n_only_backoff = 0;
 	
	int v = rnnlm.getVocabSize();
	int w = 0;

	int total_counts = 0;
	for (int i=0; i < rnnlm.getVocabSize(); i++) {
		total_counts += rnnlm.getWordCount(i);
	}

	// Initialize
	rnnlm.copyHiddenLayerToInput();

	// Initial state ( 0 | hidden layer after </s>)
	fsth.setFstHistory(rnnlm, *dzer);
	fsth.setLastWord(0);
	q.push(fsth);
	addFstState(id, new ClusterFstHistory(fsth), fst);
	fst.SetStart(INIT_STATE);
 	/*posterior.at(INIT_STATE) = MY_LOG_ONE;*/


	// Set min BO
//	min_backoff.setFstHistory(rnnlm, dzer);
	min_backoff.setLastWord(-1);
//	set_min_backoff.insert(min_backoff);
/*	for (int i=0; i < min_backoff.getNumDims(); i++) {
		min_backoff.setDim(i, 1);
	}*/
//	min_backoff = FstHistory(fsth.getNumDims(), fsth.getNumBins());
	cout << "MIN BACKOFF " << min_backoff.toString() << endl;
	

	// Final state (don't care about the associated discrete representation)
	fst.AddState();
	fst.SetFinal(FINAL_STATE, LogWeight::One());


// 	p_post = mylog(1.0
// 	             / ( (typed_dzer->getLevelSize(fsth.getNumClusters()-1))
// 	                * rnnlm.getVocabSize()
// 	               )
// 	          );
	p_post = 0.0;
	
	
	//foreach state in the queue
	while (!q.empty()) {
		fsth = q.front();
		q.pop();
		id = h2state[&fsth];
		state2h.push_back(new ClusterFstHistory(fsth));


		if (id == FINAL_STATE) { continue; }
		

 		int disc_lvl = fsth.getNumClusters()-1;
  		bo_fsth = getBackoff(rnnlm, fsth, set_min_backoff, all_prob, to_be_removed);

/*************************** NORMAL ********************************************/
		                              
  		if (fsth.getLastWord() > -1) {
  			p_post = typed_dzer->getPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized())
//  			       - typed_dzer->getWordPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized(), fsth.getLastWord());
  			       + mylog((float) rnnlm.getWordCount(fsth.getLastWord())/total_counts);
//			       + mylog(1.0/rnnlm.getVocabSize());
  		}
  		else {
			p_post = 0.0;
		}
		
// /*************************** FULL ENTROPY *******************************************/
//		entropies[disc_lvl] = 12.35686684 + log(typed_dzer->getLevelSize(disc_lvl))*0.7;
// 		//15.268085;
// 		if (entropies.find(disc_lvl) == entropies.end() && fsth.getLastWord() > -1) {
// 			entropies[disc_lvl] = 0.0;
// 			int orig_h = fsth.getFinestDiscretized();
// 			int orig_w = fsth.getLastWord();
// 			fprintf(stderr,"Computing total entropy at level %i\n", disc_lvl);
// 			for (int i = 0; i < typed_dzer->getLevelSize(disc_lvl); i++) {
// 				fsth.setDiscretized(disc_lvl, i);
// 				for (int j=0; j < rnnlm.getVocabSize(); j++) {
// 					p_post = typed_dzer->getPrior(disc_lvl, i)
// 					       - typed_dzer->getWordPrior(disc_lvl, i, j);
// //					       + mylog((float) rnnlm.getWordCount(j)/total_counts);
// //					       + mylog(1.0/rnnlm.getVocabSize());
// 					fprintf(stderr,"\rH %i\tW %i\t%f\t%f\t%.3f", i,j, typed_dzer->getWordPrior(disc_lvl, i, j), mylog((float) rnnlm.getWordCount(j)/total_counts), entropies[disc_lvl]);
// 					fsth.setLastWord(j);
// 					computeEntropyAndConditionals(entropy,
// 								      all_prob,
// 								      rnnlm,
// 								      fsth,
// 								      p_post);
// 					entropies[disc_lvl] += entropy;
// 				}
// 			}
// 			fprintf(stderr,"\n");
// 			fsth.setDiscretized(disc_lvl, orig_h);
// 			fsth.setLastWord(orig_w);
//   			p_post = typed_dzer->getPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized())
//   			       - typed_dzer->getWordPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized(), fsth.getLastWord());
// //  			       + mylog((float) rnnlm.getWordCount(fsth.getLastWord())/total_counts);
// //			       + mylog(1.0/rnnlm.getVocabSize());
// 		 	entropy = log(entropies[disc_lvl]) - 2*log(rnnlm.getVocabSize()) - log(typed_dzer->getLevelSize(disc_lvl));
// 		 	entropy = exp(entropy);
// 			//dprintf(0,"TOTAL ENTROPY(%i,%i) = %f\n",disc_lvl, fsth.getFinestDiscretized(), entropies[disc_lvl]);
// 			//dprintf(0,"AVG ENTROPY(%i,%i) = %e\n",disc_lvl, fsth.getFinestDiscretized(), entropy);
// 		}

		//dprintf(1,"-- STUDY STATE %li = %s\n", id, fsth.toString().c_str());
		//dprintf(2,"POST(%i,%i) = %f\n",disc_lvl, fsth.getFinestDiscretized(), p_post);

		computeEntropyAndConditionals(entropy,
					      all_prob,
					      rnnlm,
					      fsth,
					      p_post);
					      
					      
//		printNeurons(rnnlm.getInputLayer(),rnnlm.getVocabSize(), rnnlm.getVocabSize()+100);
		
		if (fsth.getLastWord() > -1) {
			p_w_hid = mylog((float) rnnlm.getWordCount(fsth.getLastWord())/total_counts);			
//			p_w_hid = - typed_dzer->getWordPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized(), fsth.getLastWord());
			p_hid = typed_dzer->getPrior(fsth.getNumClusters()-1, fsth.getFinestDiscretized());
			p_post = p_hid + p_w_hid;
//		 	entropy = log(entropies[disc_lvl]) - 2*log(rnnlm.getVocabSize()) - log(typed_dzer->getLevelSize(disc_lvl));
//		 	entropy = exp(entropy);
 			estimateMasses(&mass1, &mass2, rnnlm, pruning_threshold, p_post, all_prob, all_bo_prob);
		}
		else {
			mass1 = 1.0;
			mass2 = 1.0;
		}
//		entropy = entropy / rnnlm.getVocabSize();
		
//		printf("Last word = %i\n", fsth.getLastWord());
//		printNeurons(rnnlm.getInputLayer(), 10000, 10010);
//		printNeurons(rnnlm.getHiddenLayer(), 0, 9);
//		for (int i=0; i < 10; i++) {
//			rnnlm.copyHiddenLayerToInput();
//			rnnlm.computeNet(fsth.getLastWord(), 0);
//			printNeurons(rnnlm.getInputLayer(), 10000, 10010);
//			printNeurons(rnnlm.getHiddenLayer(), 0, 10);
//		}
//		printf("--\n");
		
		
		
		
		
//		mass1 = 0.9;
//		mass2 = 0.9;
		
		
		
		
		//foreach w (ie, foreach word of each class c)
		//test if the edge has to kept or removed
		backoff = false; //no backoff yet since no edge has been removed
		for (w=0; w < rnnlm.getVocabSize(); w++) {
			p = all_prob[w];
			p_joint = exp(-p_post-p);
//			p_joint = exp(-p);
//			delta = -1.0*p_joint*(-p_post-p);
//			delta = deltaProb(p, all_bo_prob[w]);
//			delta = (-1.0*exp(-p_post-p)*(-p_post-p));
//			bo_delta = (-1.0*exp(-bo_post-all_bo_prob[w])*(-bo_post-all_bo_prob[w]));
							
				
		
			delta = exp(computeDeltaEntropy(p_post,
 			                                   p,
 			                                   all_bo_prob[w],
 			                                   mass1,
 			                                   mass2)) -1.0;
  			//dprintf(1,"Delta PPL:\t%e\tMass1 = %.2f\tMass2 = %.2f\n", delta, mass1, mass2 );



			//dprintf(1,"P_cond = %e\tP_hid = %e\t\tP_w_hid = %e\tP_post = %e\tP_joint = %e\tAvg_H = %e \tDelta =%e\n",exp(-p), exp(-p_hid), exp(-p_w_hid), exp(-p_post), p_joint, entropy, delta);
//			//dprintf(1,"P = %e \tP_joint = %e \tH = %e \tDelta = %e\tBO Delta = %e\tRatio = %.2f %%\n",exp(-p), p_joint, entropy, delta, bo_delta, 100.0*((delta/entropy) / (bo_delta/bo_entropy)));


			//accept edge if this leads to a minimum
			//relative gain of the entropy
			if (fsth.getLastWord() == -1) {
//				p = mylog((float) rnnlm.getWordCount(w)/total_counts); //unigram for the minimal backoff node
				next_n_added++;
				to_be_added.push_back(w);
				to_be_added_prob.push_back(p);
				//dprintf(1,"\tACCEPT [%li] -- %i (%s) / %f --> ...\t(%e > %e)\n", id, w, rnnlm.getWordString(w), p, delta, pruning_threshold*entropy);
//				to_be_removed.push_back(w);	
			}
//			else if (delta > pruning_threshold*entropy) {
//			if (fsth.getLastWord() == -1 || (delta > (1.0-pruning_threshold))) {
			else if (delta > pruning_threshold) {
//			if (fsth.getLastWord() == -1 || (delta/entropy > pruning_threshold*(bo_delta/bo_entropy))) {
//			if (1) {
				next_n_added++;
				to_be_added.push_back(w);
				to_be_added_prob.push_back(p);
//				//dprintf(0, "ACCEPT\t%s %s\n", rnnlm.getWordString(fsth.getLastWord()), rnnlm.getWordString(w));
				//dprintf(1,"\tACCEPT [%li] -- %i (%s) / %f --> ...\t(%e > %e)\n", id, w, rnnlm.getWordString(w), p, delta, pruning_threshold*entropy);
			}
			//backoff
			else {
//				to_be_removed.push_back(w);
				backoff = true;
//				mass1 -= exp(-p);
//				mass2 -= exp(-all_bo_prob[w]);
				//dprintf(1,"\tPRUNE [%li] -- %i (%s) / %f --> ...\n", id, w, rnnlm.getWordString(w), p);
			}
			
			//print
			if (next_n_processed % 100000 == 0) {
					fprintf(stderr, "\rH=%.5f / N proc'd=%li / N added=%li (%.5f %%) / N bo=%li (%.5f %%) / %li/%li Nodes (%2.1f %%)", entropy, n_processed, n_added, ((float) n_added/ (float)n_processed)*100.0, n_backoff, ((float) n_backoff/ (float)n_added)*100.0, id, id+q.size(), 100.0 - (float) (100.0*id/(id+q.size())));
			}
			next_n_processed++;
			
//			}
		}
		
		
//		fprintf(stdout, "BILAN %s\tP-last_word = %f\tP_cluster = %f\t %.1f %% kept\n", fsth.toString().c_str(), 1.0/rnnlm.getVocabSize(), myexp(typed_dzer->getPrior(disc_lvl, fsth.getFinestDiscretized())), 100.0 * ((float) to_be_added.size() / (float) rnnlm.getVocabSize()));
		
		
		
		
		
		
		
		
		
		
		
		
		
		

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		


		//Set a part of the new FST history
		new_fsth.setFstHistory(rnnlm,*dzer);

		//if at least one word is backing off
		if (backoff) {
			
			n_backoff++;
			if (to_be_added.size() == 0) {
				n_only_backoff++;
			}
			
			



			
// 			if (bo_fsth == fsth) {
// 				printf("YEAH THE SAME %i\n", (int) set_min_backoff.size());
// 				q.push(fsth);
// 				continue;
// 			}
// 			else { printf("DIFFERENT\n"); }
			
			if (addFstState(new_id, new HierarchicalClusterFstHistory(bo_fsth), fst)) {
				q.push(bo_fsth);
				try { non_bo_pred.at(new_id) = false; }
				catch (exception e) {
					non_bo_pred.resize(new_id+(int) (non_bo_pred.size()*0.5)+1);
					non_bo_pred.at(new_id) = false;
				}
				
			}
			//dprintf(1,"BACKOFF\t[%li]\t(%s)\n-------\t[%li]\t(%s)\n", id, fsth.toString().c_str(), new_id, bo_fsth.toString().c_str());

			fst.AddArc(id, LogArc(EPSILON, EPSILON, LogWeight::Zero(), new_id));
			
			addPred(pred, new_id, id);
			
		}
		
		
		vector<real>::iterator it_p = to_be_added_prob.begin();
		for (vector<int>::iterator it = to_be_added.begin(); it != to_be_added.end(); ++it) {
			w = *it;
			p = *it_p;

			if (w == 0) {
				fst.AddArc(id, LogArc(FstWord(w),FstWord(w),p,FINAL_STATE));
				//dprintf(1,"EDGE [%li] (%s)\n---- %i (%s) / %f -->\n---- [%li] FINAL STATE)\n\n", id, fsth.toString().c_str(), FstWord(w), rnnlm.getWordString(w), p, FINAL_STATE);				
			}
		
			//accept edge
			else {
				new_fsth.setLastWord(w);
	
				//if sw not in the memory
				//then add a new state for sw in the FST and push sw in the queue
				if (addFstState(new_id, new HierarchicalClusterFstHistory(new_fsth), fst)) {
					q.push(new_fsth);
					try { non_bo_pred.at(new_id) = true; }
					catch (exception e) {
						non_bo_pred.resize(new_id+(int) (non_bo_pred.size()*0.5)+1);
						non_bo_pred.at(new_id) = true;
					}
				}
				else { /* already exists */ }
			
				//add the edge in the FST
				non_bo_pred.at(new_id) = true;
				fst.AddArc(id, LogArc(FstWord(w),FstWord(w),p,new_id));
				//dprintf(1,"EDGE [%li] (%s)\n---- %i (%s) / %f -->\n---- [%li] (%s)\n\n", id, fsth.toString().c_str(), FstWord(w), rnnlm.getWordString(w), p, new_id, new_fsth.toString().c_str());				

//				posterior.at(new_id) += posterior[id]*p;

			}
			
			++it_p;
		}
		
		n_added = next_n_added;
		n_processed = next_n_processed;
		
		//reset queues
		to_be_added.clear();
		to_be_added_prob.clear();
//		to_be_removed.clear();
		
	}

	cout << endl;
	
	//compute backoff weights
	deleted = compactBackoffNodes(fst, pred, non_bo_pred);
	computeAllBackoff(fst, pred);


	//remove useless nodes
	removeStates(fst, new_fst, deleted);
	fst.DeleteStates();
	fst = new_fst;
	
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




