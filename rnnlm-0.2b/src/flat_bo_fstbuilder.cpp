///////////////////////////////////////////////////////////////////////
//
// Specialization of FstBuilder using a different backoff strategy
// based on the search for a flatest distribution for backoff nodes
// instead of the min KL divergence for the classical builder
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////


#include "flat_bo_fstbuilder.h"


// /**
//  * Return the backoff FST state for a given backed off FST state
//  * and eventually update the set of minimal backoff nodes
//  */
// FstHistory FlatBOFstBuilder::getBackoff(CRnnLM &rnnlm,
//                       const FstHistory &fsth,
//                       set<FstHistory> &set_min_bo,
//                       vector<real> &cur_cond,
//                       vector<int> &words)
// {
// 	//First test if fsth is a min BO node
// 	if (set_min_bo.find(fsth) != set_min_bo.end()) {
// 		return fsth;
// 	}
// 
// 	// Compute number of steps
// 	float ratio=1.0;
// 	int n_bo_loops = (int) (ratio* 
// 	                             ((1+rnnlm.getHiddenLayerSize())*(dzer.getNumBins()-1))
// 	                             / max_backoff_path);
// 	n_bo_loops++;
// 	int steps = n_bo_loops;
// 	dprintf(1,"\nactual step is %i\n", steps);
// 	
// 	
// 	// Copy the current FST history
// 	FstHistory bo(fsth);
// 	FstHistory cand = FstHistory(fsth);
// 	vector<dist_dim_val_triple> distances;
// 	dist_dim_val_triple dd;
// 	int x = 0;
// 
// 	real dist = 0.0;
// 
// 	real uniform_logprob = mylog(1.0/rnnlm.getVocabSize());
// 	vector<FstHistory> candidates;
// 	vector<real> bo_cond(rnnlm.getVocabSize());
// 	vector<real> uniform_cond(rnnlm.getVocabSize(), uniform_logprob);
// 	real self_distance = distanceKL(uniform_cond, cur_cond);
// 	dprintf(2,"SELF\t%f\n", self_distance);	
// 
// 	
// 	//browse all dimensions
// 	for (int i=0; i < rnnlm.getHiddenLayerSize(); i++) {
// 		for (x = 0;
// 		     x < dzer.getNumBins();
// 		     x++)
// 		{
// 			if (fsth.getDim(i) != x) {
// 
// 				// Change a bit
// 				cand.setDim(i,x);
// 		
// 				//Compute L2 distance between the 2 conditional distributions
// 				bo_cond = computeSomeConditionals(rnnlm, cand, words);
// 				
// 	//			dist = distanceL2(tronc_cur_cond, bo_cond);
// 				dist = distanceKL(uniform_cond, bo_cond);
// 				dd = dist_dim_val_triple(dist, std::pair<int,int>(i,x) );
// 				distances.push_back(dd);
// 				dprintf(2,"DIM %i with X %i\t%f\n", i, x, dist);
// //	printf("CAND %i is %s\n", i, cand.toString().c_str());		
// //	printf("     dist is %f\n", dist);		
// 				// Restore to the current history
// 				cand.setDim(i,fsth.getDim(i));
// 			}
// 		}
// 	}
// 
// 	//sort
// 	std::sort(distances.begin(), distances.end());
// 	for (int i=0; i < distances.size() && i < steps; i++) {
// 		dprintf(2,"dim %i = %i\t%f\n", distances[i].second.first, distances[i].second.second, distances[i].first);
// 	}
// 	//if the best bo candidate is worse than the current node
// 	//then the current node is a 
// 	if (distances[0].first > self_distance) {
// //		dprintf(1,"NEW MIN BO:\t%s\n", fsth.toString().c_str());
// //		set_min_bo.insert(fsth);
// 		return fsth;
// 	}
// 	
// 	for (int i=0; i < distances.size() && distances[i].first < self_distance && i < steps; i++) {
// 		std::pair<int,int> change = distances[i].second;
// 		int dim = change.first;
// 		int val = change.second;
// 		if (abs(val - bo.getDim(dim)) <= (steps - i)) {
// 			dprintf(2,"CHANGE BO DIM %i = %i\t%f\n", distances[i].second.first, distances[i].second.second, distances[i].first);
// 			steps -= (abs(val - bo.getDim(dim))-1);
// 			bo.setDim(dim,val);
// 		}
// 	}
// 	
// 	
// //			printf("src    : %s\nbo     : %s\nmin_bo: %s\n",fsth.toString().c_str(), bo.toString().c_str(), min_bo.toString().c_str());	
// //	printf("MIN DIM is %i\n", min_dim);
// //	dprintf("MIN STATE is %s\n", bo.toString().c_str());
// //	printf("COMPARED TO  %s\n", fsth.toString().c_str());
// 	bo_cond = computeSomeConditionals(rnnlm, bo, words);
// 				
// 	//			dist = distanceL2(tronc_cur_cond, bo_cond);
// 	dist = distanceKL(uniform_cond, bo_cond);
// 	dprintf(2,"FINAL BO DISTANCE IS\t%f\n", dist);
// 	return bo;
// 
// }
// 	


/**
 * Return the backoff FST state for a given backed off FST state
 * and eventually update the set of minimal backoff nodes
 */
NeuronFstHistory FlatBOFstBuilder::getBackoff(CRnnLM &rnnlm,
                      const NeuronFstHistory &fsth,
                      set<NeuronFstHistory> &set_min_bo,
                      vector<real> &cur_cond,
                      vector<int> &words)
{
	//First test if fsth is a min BO node
	if (set_min_bo.find(fsth) != set_min_bo.end()) {
		return fsth;
	}

	// Compute number of steps
	float ratio=1.0;
	int n_bo_loops = (int) (ratio* 
	                             ((1+rnnlm.getHiddenLayerSize())*(getNumBins()-1))
	                             / max_backoff_path);
	n_bo_loops++;
	int steps = n_bo_loops;
	dprintf(1,"\nactual step is %i\n", steps);
	
	
	// Copy the current FST history
	NeuronFstHistory bo(fsth);
	NeuronFstHistory cand = NeuronFstHistory(fsth);
	int x = 0;

	real dist = 0.0;

	real uniform_logprob = mylog(1.0/rnnlm.getVocabSize());
	vector<FstHistory> candidates;
	vector<real> bo_cond(rnnlm.getVocabSize());
	vector<real> uniform_cond(rnnlm.getVocabSize(), uniform_logprob);
	real self_distance = distanceKL(uniform_cond, cur_cond);
	dprintf(2,"SELF\t%s\t%f\n", fsth.toString().c_str(), self_distance);	

	for (int s=0; s < steps; s++) {
		int min_dim = 0;
		int min_val = 0;
		real min_dist = 1e10;
		
		//browse all dimensions
		for (int i=0; i < rnnlm.getHiddenLayerSize(); i++) {
			for (x = max(0,bo.getDim(i)-1); x <= min(getNumBins()-1, bo.getDim(i)+1); x++) {
				if (bo.getDim(i) != x) {

					// Change a bit
					cand.setDim(i,x);
		
					//Compute L2 distance between the 2 conditional distributions
					bo_cond = computeSomeConditionals(rnnlm, cand, words);
		//			dist = distanceL2(tronc_cur_cond, bo_cond);
					dist = distanceKL(uniform_cond, bo_cond);
					dprintf(2,"DIM %i with X %i\t%f\n", i, x, dist);
	//	printf("CAND %i is %s\n", i, cand.toString().c_str());		
	//	printf("     dist is %f\n", dist);	
		
					// Restore to the current history
					cand.setDim(i,bo.getDim(i));
					
					//update if better
					if (dist < min_dist) {
						min_dist = dist;
						min_dim = i;
						min_val = x;
					}
				}
			}
		}
		
		//if the best bo candidate is worse than the current node
		//then the current node is a 
		if (min_dist > self_distance) {
			break;
		}
	
		dprintf(2,"CHANGE BO DIM %i = %i\tHIST %s\tDIST %f\n", min_dim, min_val, bo.toString().c_str(), min_dist);
		bo.setDim(min_dim,min_val);
		self_distance = min_dist;
		
	}
	
//			printf("src    : %s\nbo     : %s\nmin_bo: %s\n",fsth.toString().c_str(), bo.toString().c_str(), min_bo.toString().c_str());	
//	printf("MIN DIM is %i\n", min_dim);
//	dprintf("MIN STATE is %s\n", bo.toString().c_str());
//	printf("COMPARED TO  %s\n", fsth.toString().c_str());
	bo_cond = computeSomeConditionals(rnnlm, bo, words);
				
	//			dist = distanceL2(tronc_cur_cond, bo_cond);
	dist = distanceKL(uniform_cond, bo_cond);
	dprintf(2,"FINAL BO DISTANCE IS\t%f\n", dist);
	return bo;

}
	





/**
 * Create an FST based on an RNN
 */
void FlatBOFstBuilder::convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) {
	queue<NeuronFstHistory> q;
	VectorFst<LogArc> new_fst;
	
	NeuronFstHistory fsth(rnnlm.getHiddenLayerSize(),getNumBins());
	FstIndex id = 0;
	
	NeuronFstHistory new_fsth(rnnlm.getHiddenLayerSize(),getNumBins());
	FstIndex new_id;

	NeuronFstHistory min_backoff(rnnlm.getHiddenLayerSize(),getNumBins());
	set<NeuronFstHistory>set_min_backoff;
	
	NeuronFstHistory bo_fsth(rnnlm.getHiddenLayerSize(),getNumBins());
	bool backoff = false;
	vector<FstIndex> deleted;


	real p = 0.00;
	real p_joint = 0.00;
	real entropy = 0.0;
	real delta = 0.0;
	vector<real> all_prob(rnnlm.getVocabSize());
 	vector<real> posterior(10);
	
	map< FstIndex,set<FstIndex> > pred;
	vector<bool> non_bo_pred(rnnlm.getVocabSize());
	vector<int> to_be_added;
	vector<int> to_be_removed;
	for (int i = 0; i < rnnlm.getVocabSize(); i++) {
		to_be_removed.push_back(i);
	}
	vector<real> to_be_added_prob;


 	FstIndex n_added = 0;
 	FstIndex n_processed = 0;
 	FstIndex next_n_added = 0;
 	FstIndex next_n_processed = 0;
 	FstIndex n_backoff = 0;
 	FstIndex n_only_backoff = 0;
 	
	int v = rnnlm.getVocabSize();
	int w = 0;


	// Initialize
	rnnlm.copyHiddenLayerToInput();
//	printNeurons(rnnlm.getInputLayer(),0,10);

	// Initial state ( 0 | hidden layer after </s>)
	printNeurons(rnnlm.getHiddenLayer(),0,10);
	fsth.setFstHistory(rnnlm, *dzer);
	fsth.setLastWord(0);
	q.push(fsth);
	addFstState(id, new NeuronFstHistory(fsth), fst);
	fst.SetStart(INIT_STATE);
	
	// Final state (don't care about the associated discrete representation)
	fst.AddState();
	fst.SetFinal(FINAL_STATE, LogWeight::One());
	
 	/*posterior.at(INIT_STATE) = MY_LOG_ONE;*/
	min_backoff.setLastWord(-1);
	computeEntropyAndConditionals(entropy, all_prob, rnnlm, min_backoff);
	min_backoff = getBackoff(rnnlm, min_backoff, set_min_backoff, all_prob, to_be_removed);
	cout << "MIN BACKOFF " << min_backoff.toString() << endl;
	set_min_backoff.insert(min_backoff);
	
//	addFstState(id, min_backoff, fst);
//	q.push(min_backoff);
	

	
	// Estimate number of backoff loop to bound the backoff path length
// 	float ratioa = 0.0;
// 	float ratiob = 0.0;
	float ratio = 0.0;
// 	for (int i=0; i < min_backoff.getNumDims(); i++) {
// 		if (min_backoff.getDim(i) == 1) {
// 			ratioa++;
// 		}
// 		if (fsth.getDim(i) == 1) {
// 			ratiob++;
// 		}
// 	}
// 	ratioa /= min_backoff.getNumDims();
// 	ratiob /= min_backoff.getNumDims();
// 	ratio = (ratioa*(1.0-ratiob))+(ratiob*(1.0-ratioa));
	ratio=1.0;

//	printf("ratio=%f\t%i BO loops\n", ratio, n_bo_loops);
	
	
	
	//foreach state in the queue
	while (!q.empty()) {
		fsth = q.front();
		q.pop();
		id = h2state[&fsth];
		state2h.push_back(new NeuronFstHistory(fsth));
		if (id == FINAL_STATE) { continue; }


		
		
	dprintf(1,"-- STUDY STATE %li = %s\n", id, fsth.toString().c_str());
	

/*		try { posterior.at(id) = MY_LOG_ONE; }
		catch (exception e) {
			posterior.resize((int) (posterior.size()*1.5)+1);
			posterior.at(id) = MY_LOG_ONE;
		}*/
		
		computeEntropyAndConditionals(entropy, all_prob, rnnlm, fsth);
		
		//compute BO in advance and check if it is a min BO node
		bo_fsth = getBackoff(rnnlm, fsth, set_min_backoff, all_prob, to_be_removed);
		if (bo_fsth == fsth) { bo_fsth = min_backoff; }
			
		//foreach w (ie, foreach word of each class c)
		//test if the edge has to kept or removed
		backoff = false; //no backoff yet since no edge has been removed
		for (w=0; w < rnnlm.getVocabSize(); w++) {
				p = all_prob[w];
				
				/*p_joint = exp(-posterior[id]-p);*/
				p_joint = exp(-p);
				delta = -1.0*p_joint*log2(p_joint);
				
				//accept edge if this leads to a minimum
				//relative gain of the entropy

				dprintf(2,"P = %e \tP_joint = %e \tH = %e \tDelta =%e \tDelta H = %.6f %%\n",exp(-p), p_joint, entropy, delta, 100.0*delta/entropy);

				if (set_min_backoff.find(fsth) != set_min_backoff.end() || (delta > pruning_threshold*entropy)) {
//				if ((fsth == min_backoff) || (delta > pruning_threshold*entropy)) {
					next_n_added++;
					to_be_added.push_back(w);
					to_be_added_prob.push_back(p);
					dprintf(2,"\tACCEPT [%li] -- %i (%s) / %f --> ...\t(%e > %e)\n", id, w, rnnlm.getWordString(w), p, delta, pruning_threshold*entropy);
//					to_be_removed.push_back(w);
 				}
 				//backoff
				else {
//					to_be_removed.push_back(w);
					backoff = true;
					dprintf(2,"\tPRUNE [%li] -- %i / %f --> ...\n", id, w, p);
 				}
 				
 				//print
				if (next_n_processed % 100000 == 0) {
						fprintf(stderr, "\rH=%.5f / N proc'd=%li / N added=%li (%.5f %%) / N bo=%li (%.5f %%) / %li/%li Nodes (%2.1f %%) / N min BO=%i", entropy, n_processed, n_added, ((float) n_added/ (float)n_processed)*100.0, n_backoff, ((float) n_backoff/ (float)n_added)*100.0, id, id+q.size(), 100.0 - (float) (100.0*id/(id+q.size())), (int) set_min_backoff.size());
				}
				next_n_processed++;
 				
//			}
		}


		//Set a part of the new FST history
		new_fsth.setFstHistory(rnnlm, *dzer);

		//if at least one word is backing off
		if (backoff) {
			
			n_backoff++;
			if (to_be_added.size() == 0) {
				n_only_backoff++;
			}
			
			
			if (addFstState(new_id, new NeuronFstHistory(bo_fsth), fst)) {
				q.push(bo_fsth);
				try { non_bo_pred.at(new_id) = false; }
				catch (exception e) {
					non_bo_pred.resize(new_id+(int) (non_bo_pred.size()*0.5)+1);
					non_bo_pred.at(new_id) = false;
				}
				
			}
			dprintf(1,"BACKOFF\t[%li]\t(%s)\n-------\t[%li]\t(%s)\n", id, fsth.toString().c_str(), new_id, bo_fsth.toString().c_str());

			fst.AddArc(id, LogArc(EPSILON, EPSILON, LogWeight::Zero(), new_id));
			
			addPred(pred, new_id, id);
			
		}
		
		
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
				if (addFstState(new_id, new NeuronFstHistory(new_fsth), fst)) {
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
				dprintf(1,"EDGE [%li] (%s)\n---- %i (%s) / %f -->\n---- [%li] (%s)\n\n", id, fsth.toString().c_str(), FstWord(w), rnnlm.getWordString(w), p, new_id, new_fsth.toString().c_str());				

//				posterior.at(new_id) += posterior[id]*p;

			}
			
			/*if (posterior[id]+p < LogWeight::Zero().Value()) {
				p_joint = exp(-posterior[id]-p);
				entropy -= p_joint*log2(p_joint);
			}*/
			
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

						//printf("H=%.5f / N proc'd=%li / N added=%li (%.5f %%) %li/%li Nodes (%2.1f %%)\n", entropy, n_processed, n_added, ((float) n_added/ (float)n_processed)*100.0, id, id+q.size(), 100.0 - (float) (100.0*id/(id+q.size())));
	cout << "END" << endl;
	
}








