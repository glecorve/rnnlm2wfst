///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////


#include "neuron_fstbuilder.h"



/**
 * Return the backoff FST state for a given backed off FST state
 * and eventually update the set of minimal backoff nodes
 */
NeuronFstHistory NeuronFstBuilder::getBackoff(CRnnLM &rnnlm,
                      const NeuronFstHistory &fsth,
                      set<NeuronFstHistory> &set_min_bo,
                      vector<real> &cur_cond,
                      vector<int> &words)
{

	// Compute number of steps
	set<NeuronFstHistory>::iterator bo_it = set_min_bo.begin();
	
	const NeuronFstHistory &min_bo = *bo_it;
	float ratio=1.0;
	int n_bo_loops = (int) (ratio* 
	                             ((1+rnnlm.getHiddenLayerSize())*(getNumBins()-1))
	                             / max_backoff_path);
	n_bo_loops++;
	
	int fsth_dist = fsth.distanceL1(min_bo);
	int steps = (fsth_dist % (n_bo_loops));
	if (steps == 0 && fsth_dist != 0) {
		steps = n_bo_loops;
	}
	dprintf(1,"\ndist is %i / normal step is %i / actual step is %i\n", fsth_dist, n_bo_loops, steps);

	// Copy the current FST history
	NeuronFstHistory bo(fsth);
	NeuronFstHistory cand = NeuronFstHistory(fsth);
	vector<dist_dim_pair> distances;
	dist_dim_pair dd;
	int x = 0;

	real dist = 0.0;

	if (bo.sameDiscretization(&min_bo)) {
		return min_bo;
	}
	
	vector<NeuronFstHistory> candidates;
	vector<real> bo_cond(rnnlm.getVocabSize());
	vector<real> tronc_cur_cond(rnnlm.getVocabSize(), MY_LOG_ZERO);
	
	for (int i = 0; i < words.size(); i++) {
		tronc_cur_cond[words[i]] = cur_cond[words[i]];
	}
	

	//browse all dimensions
	for (int i=0; i < rnnlm.getHiddenLayerSize(); i++) {
		for (x = min(fsth.getDim(i),min_bo.getDim(i));
		     x <= max(fsth.getDim(i),min_bo.getDim(i));
		     x++)
		{
			if (fsth.getDim(i) != x) {
		
				// Change a bit
				cand.setDim(i,x);
		
				//Compute L2 distance between the 2 conditional distributions
				bo_cond = computeSomeConditionals(rnnlm, cand, words);
				
	//			dist = distanceL2(tronc_cur_cond, bo_cond);
				dist = distanceKL(tronc_cur_cond, bo_cond);
				dd = dist_dim_pair(dist,i);
				distances.push_back(dd);
	//			printf("DIM %i with X %i\t%f\n", i, x, dist);
//	printf("CAND %i is %s\n", i, cand.toString().c_str());		
//	printf("     dist is %f\n", dist);		
				// Restore to the current history
				cand.setDim(i,fsth.getDim(i));
			}
		}
	}

	//sort
	std::sort(distances.begin(), distances.end());
	if (distances.size() < steps) {
		return min_bo;
	}
	for (int i=0; i < distances.size() && i < steps; i++) {
		int d = distances[i].second;
		if (bo.getDim(d) > min_bo.getDim(d)) {
			bo.setDim(d,bo.getDim(d)-1);
		}
		else if (bo.getDim(d) < min_bo.getDim(d)) {
			bo.setDim(d,bo.getDim(d)+1);
		}
	}
	
	
//			printf("src    : %s\nbo     : %s\nmin_bo: %s\n",fsth.toString().c_str(), bo.toString().c_str(), min_bo.toString().c_str());	
//	printf("MIN DIM is %i\n", min_dim);
//	printf("MIN STATE is %s\n", bo.toString().c_str());
//	printf("COMPARED TO  %s\n", fsth.toString().c_str());
	return bo;
}






/**
 * Compute the conditional probability of a word given a state.
 * This computation handles backoffs in the FST.
 */
real NeuronFstBuilder::computeFstWordProb(VectorFst<LogArc> &fst, int word, FstIndex state) {
	Matcher< VectorFst<LogArc> > matcher(fst, MATCH_INPUT);
	matcher.SetState(state);
	LogWeight prob = LogWeight::One();
	while (!matcher.Find(word)) {
		ArcIterator< VectorFst<LogArc> > it(matcher.GetFst(), state);
		
		prob = Times(prob,it.Value().weight); //apply backoff weight
//		printf("\t\t%i (%li/eps) --> %i = %f\t\tEPS\n", it.Value().ilabel , state, it.Value().nextstate, it.Value().weight.Value());
		state = it.Value().nextstate;
		matcher.SetState(state);
	}
	prob = Times(prob, matcher.Value().weight);
//	printf("\t\t%i (%li/w%i) --> %i = %f\n", matcher.Value().ilabel , state, word, matcher.Value().nextstate, matcher.Value().weight.Value());
	return prob.Value();
}











/**
 * Compute the backoff weight for one backoff edge between src and bo
 * weight = 1 - sum_{w in A} P(w|src) / 1 - sum_{w in A} P(w|bo)
 * where A is the set of word whose edge hasn't been pruned in src
 */
void NeuronFstBuilder::computeOneBackoff(VectorFst<LogArc> &fst, FstIndex src, FstIndex bo) {
	real mass_normal = 0.0;
	real mass_bo = 0.0;
	
	//browse all non backoff edges from src
	printf("BACKOFF FOR NODE %li\n", src);
	MutableArcIterator<VectorFst<LogArc> > aiter(&fst, src);
	for (; !aiter.Done(); aiter.Next()) {
		const LogArc &src_arc = aiter.Value();
		if (src_arc.ilabel == EPSILON) { continue; } //skip epsilon edge
			dprintf(0,"\tSRC:\tw%i\t%f\n", src_arc.ilabel, src_arc.weight.Value());
			dprintf(0,"\tBCK:\tw%i\t%f\n", src_arc.ilabel, computeFstWordProb(fst, src_arc.ilabel, bo));
			mass_normal += myexp(src_arc.weight.Value());
			mass_bo += myexp(computeFstWordProb(fst, src_arc.ilabel, bo));
	}
	
	dprintf(0,"Backoff = ((1 - %f)/(1 - %f)) = %f\n", mass_normal, mass_bo, ((1 - mass_normal)/(1 - mass_bo)));
	aiter.Reset(); //go back to the backoff arc, ie, the first arc.
	aiter.SetValue(LogArc(EPSILON,EPSILON, mylog(1 - mass_normal) - mylog(1 - mass_bo), bo));
	printf("\n\n");
}













/**
 * Compute the backoff weight for each backoff edge
 */
void NeuronFstBuilder::computeAllBackoff(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred) {
	
	set<FstIndex>::iterator itv;
	set<FstIndex> v;
	int n_bo = pred.size();
	int i = 0;
		printf("Start of BO computation\n");
// 	map< FstIndex,vector<FstIndex> >::reverse_iterator itm;
// 	for(itm = pred.rbegin(); itm != pred.rend(); ++itm) {
 	map< FstIndex,set<FstIndex> >::iterator itm;
 	for(itm = pred.begin(); itm != pred.end(); ++itm) {
		v = itm->second;
		for (itv = v.begin(); itv != v.end(); ++itv) {
//			cout << itm->first << " <-bo- " << *itv << endl;
			computeOneBackoff(fst, *itv, itm->first);
		}
		i++;
		if (i % 1000 == 0) {
			dprintf(1,"\rBackoff = %i nodes / %i\t(%.6f %%)\t[ %.2f %% BO nodes proc'd ]", i, n_bo, (float) (100.0*i/n_bo), ((float) 100.0*(i/pred.size())) );
		}
	}
		printf("End of BO computation\n");
	cout << endl;
}












/**
 * Change destination of every edge arriving in "old_target" and replace with "new_target"
 */
void NeuronFstBuilder::changeTargetForNode(VectorFst<LogArc> &fst,
                         FstIndex old_target,
                         FstIndex new_target) {
	for (StateIterator<VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
		StateId state = siter.Value();
		for (MutableArcIterator<VectorFst<LogArc> > aiter(&fst, state); !aiter.Done(); aiter.Next()) {
			const LogArc &arc = aiter.Value();
			if (arc.nextstate == old_target) {
				dprintf(2,"\tMove %li --[%i / %f ]--> %li \tto\t%lu\n", state, arc.ilabel, arc.weight.Value(), old_target, new_target);
				 aiter.SetValue(LogArc(arc.ilabel, arc.olabel, arc.weight.Value(), new_target));
			}
		}
	}
}










/**
 * Remove a FST state ID "src" from the list of predecessors of FST state ID "dest"
 */
void NeuronFstBuilder::removePred(map< FstIndex,vector<FstIndex> > &pred, FstIndex dest, FstIndex src) {
	vector<FstIndex>::iterator it;
	for (it = pred[dest].begin(); it != pred[dest].end(); ++it) {
		if (*it == src) {
			it = pred[dest].erase(it);
		}
	}
}









void NeuronFstBuilder::removeStates(const VectorFst<LogArc> &old_fst, VectorFst<LogArc> &new_fst, vector<FstIndex> &to_be_deleted) {
	map<StateId,StateId> new_id;
	StateId shift = 0;
	sort(to_be_deleted.begin(), to_be_deleted.end());
	
	//compute new ids
	for (StateId i=0; i < old_fst.NumStates(); i++) {
		if (shift == to_be_deleted.size() || i != to_be_deleted[shift]) {
			dprintf(1,"KEEP NODE %li\n",i);
			new_id[i] = new_fst.AddState();
			if (i > 1) {
				printf("NODE %li\t\t%s\n", new_id[i], state2h[i-1]->toString().c_str());
			}
			else if (i == 0) {
				printf("NODE 0\t\t%s\n", state2h[i]->toString().c_str());
			}
			else {
				printf("NODE 1\t\tFINAL (SINK) NODE\n");
			}
			
			if (old_fst.Start() == i) {
				new_fst.SetStart(new_id[i]);
			}
		}
		else {
			shift++;
		}
	}
	
	
	//copy undeleted states and replace ids on arcs
	shift = 0;
	for (StateIterator<VectorFst<LogArc> > siter(old_fst);
	     !siter.Done();
	     siter.Next())
	{
		StateId state = siter.Value();
		if (shift == to_be_deleted.size() || state != to_be_deleted[shift]) {
			for (ArcIterator<VectorFst<LogArc> > aiter(old_fst, state);
			     !aiter.Done();
			     aiter.Next())
			{
				const LogArc &arc = aiter.Value();
				
 				new_fst.AddArc(new_id[state], LogArc(arc.ilabel, arc.olabel, arc.weight.Value(), new_id[arc.nextstate]));
//				new_fst.AddArc(new_id[state], LogArc(arc.ilabel, arc.olabel, exp(-arc.weight.Value()), new_id[arc.nextstate]));
				dprintf(2,"NEW ARC: %li\t%i\t%f\t%li\n", new_id[state], arc.ilabel, exp(-arc.weight.Value()), new_id[arc.nextstate]);
			}
			
			//duplicate final weight
			if (old_fst.Final(state) != LogWeight::Zero()) {
				new_fst.SetFinal(new_id[state], old_fst.Final(state).Value());
			}
		}
		else {
			shift++;
		}
	}
	return;	
}














/**
 * Removes useless nodes, nodes with only a backoff output
 * and plug their input to this single output
 */
vector<FstIndex> NeuronFstBuilder::compactBackoffNodes(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred, vector<bool> &non_bo_pred) {
	set<FstIndex>::iterator itv;
	set<FstIndex> v;
	map<FstIndex,FstIndex> redir;
	vector<FstIndex> deleted;
 	map< FstIndex,set<FstIndex> >::iterator itm;
 	for(itm = pred.begin(); itm != pred.end(); ++itm) {
		MutableArcIterator<VectorFst<LogArc> > bo_node(&fst, itm->first);
		v = itm->second;
		for (itv = v.begin(); itv != v.end(); ++itv) {
		//if only one output (the backoff edge)
			if (fst.NumArcs(*itv) == 1) {
				//cout << "REDIR " << *itv << " TO " << itm->first << endl;
				if (*itv != itm->first) {
					redir[*itv] = itm->first;
				}
				dprintf(2,"DIRECT REDIR %li TO %li\n", *itv, itm->first);
			}
		}
	}
	

	FstIndex source;
	FstIndex target;
	map<FstIndex,FstIndex>::iterator transitivity;	
	for (map<FstIndex,FstIndex>::iterator it_redir = redir.begin(); it_redir != redir.end(); ++it_redir) {
		source = it_redir->first;
		//try to find the final target, eg, target(a) = c whem a->b and b->c
		target = it_redir->second;
		transitivity = redir.find(target);
		while (transitivity != redir.end()) {
			target = transitivity->second;
			transitivity = redir.find(target);
		}
		if (target == source) { continue; }
		
		dprintf(1,"REDIR %li TO %li\n", source, target);
		
		//change FST
		changeTargetForNode(fst, source, target);
		deleted.push_back(source); //mark as "to be deleted"
		if (source == INIT_STATE) {
			fst.SetStart(target);
		}
		//change map of backoffs
		for (set<FstIndex>::iterator it = pred[source].begin();
		     it != pred[source].end();
		     ++it) {
		    map<FstIndex,FstIndex>::iterator it_find = redir.find(*it);
		    if (it_find == redir.end()) {
		    	dprintf(2,"\tAdd %li to bo list of %li\n", *it, target);
				pred[target].insert(*it);
			}
		}
		for (set<FstIndex>::iterator it = pred[target].begin();
		     it != pred[target].end();
		     ++it) {
		    if (*it == source) {
		    	dprintf(2,"\tRemove %li from bo list of %li\n", source, target);
				pred[target].erase(it);
			}
		}
		pred.erase(source);
		
	}
		printf("End of compaction\n");

//	fst.DeleteStates(deleted);
	return deleted;
	
}











/* ========================================================================================================
                                            CONVERTION METHOD
   ======================================================================================================== */







/**
 * Create an FST based on an RNN
 */
void NeuronFstBuilder::convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) {
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
	fsth.setFstHistory(rnnlm, *dzer);
	fsth.setLastWord(0);
	q.push(fsth);
	addFstState(id, new NeuronFstHistory(fsth), fst);
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

				dprintf(1,"P = %e \tP_joint = %e \tH = %e \tDelta =%e \tDelta H = %.6f %%\n",exp(-p), p_joint, entropy, delta, 100.0*delta/entropy);

				if (set_min_backoff.find(fsth) != set_min_backoff.end() || (delta > pruning_threshold*entropy)) {
//				if ((fsth == min_backoff) || (delta > pruning_threshold*entropy)) {
					next_n_added++;
					to_be_added.push_back(w);
					to_be_added_prob.push_back(p);
					dprintf(1,"\tACCEPT [%li] -- %i (%s) / %f --> ...\t(%e > %e)\n", id, w, rnnlm.getWordString(w), p, delta, pruning_threshold*entropy);
					to_be_removed.push_back(w);
 				}
 				//backoff
				else {
					to_be_removed.push_back(w);
					backoff = true;
					dprintf(1,"\tPRUNE [%li] -- %i / %f --> ...\n", id, w, p);
 				}
 				
 				//print
				if (next_n_processed % 100000 == 0) {
						fprintf(stderr, "\rH=%.5f / N proc'd=%li / N added=%li (%.5f %%) / N bo=%li (%.5f %%) / %li/%li Nodes (%2.1f %%)", entropy, n_processed, n_added, ((float) n_added/ (float)n_processed)*100.0, n_backoff, ((float) n_backoff/ (float)n_added)*100.0, id, id+q.size(), 100.0 - (float) (100.0*id/(id+q.size())));
				}
				next_n_processed++;
 				
//			}
		}


		//Set a part of the new FST history
		new_fsth.setFstHistory(rnnlm,*dzer);

		//if at least one word is backing off
		if (backoff) {
			
			n_backoff++;
			if (to_be_added.size() == 0) {
				n_only_backoff++;
			}
			
			


			bo_fsth = getBackoff(rnnlm, fsth, set_min_backoff, all_prob, to_be_removed);
			
			if (bo_fsth == fsth) {
				printf("YEAH THE SAME %i\n", (int) set_min_backoff.size());
				q.push(fsth);
				continue;
			}
			else { printf("DIFFERENT\n"); }
			
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
		to_be_removed.clear();
		
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





