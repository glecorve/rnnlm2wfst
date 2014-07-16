///////////////////////////////////////////////////////////////////////
//
// Converts a recurrent neural network into a finite state transducer
// in order to integrate long-span information within the decoding process
//
// Gwénolé Lecorvé
// Oct. 2011
//
///////////////////////////////////////////////////////////////////////


#include "hierarchical_cluster_fstbuilder.h"



/**
 * Return the backoff FST state for a given backed off FST state
 * and eventually update the set of minimal backoff nodes
 */
HierarchicalClusterFstHistory HierarchicalClusterFstBuilder::getBackoff(
                      CRnnLM &rnnlm,
                      const HierarchicalClusterFstHistory &fsth,
                      set<HierarchicalClusterFstHistory> &set_min_bo,
                      vector<real> &cur_cond,
                      vector<int> &words)
{

	HierarchicalClusterFstHistory bo(fsth);
	if (bo.getNumClusters() == 1) {
		bo.setLastWord(-1);
		return bo;
	}
	else {
		bo.reduceDiscretization();
	}
	
	return bo;
}


// 
// /**
//  * Return the backoff FST state for a given backed off FST state
//  * and eventually update the set of minimal backoff nodes
//  */
// HierarchicalClusterFstHistory HierarchicalClusterFstBuilder::getBackoff(
//                       CRnnLM &rnnlm,
//                       const HierarchicalClusterFstHistory &fsth,
//                       HierarchicalClusterDiscretizer &dzer,
//                       vector<real> &cur_cond,
//                       vector<int> &words)
// {
// 
// 	HierarchicalClusterFstHistory bo(fsth);
// 	vector<real> bo_cond(cur_cond.size());
// 	int k = fsth.getFinestDiscretized();
// 	float kl_div = 0.0;
// 	int min_k = k;
// 	float min_kl_div = 10000.0;
// 	int lvl = fsth.getNumClusters()-1;
// 	for (int i=0; i<dzer.getLevelSize(lvl); i++) {
// 		if (i != k && dzer.getPrior(lvl,i) > dzer.getPrior(lvl,k)) {
// 			bo.setDiscretized(lvl,i);
// 			bo_cond = computeAllConditionals(rnnlm, bo);
// 			kl_div = distanceKL(cur_cond, bo_cond);
// 			if (kl_div < min_kl_div) {
// 				min_kl_div = kl_div;
// 				min_k = i;
// 			}
// 		}
// 	}
// 	
// 	if (lvl > 0) {
// 		bo.reduceDiscretization();
// 		bo_cond = computeAllConditionals(rnnlm, bo);
// 		kl_div = distanceKL(cur_cond, bo_cond);
// 		if (kl_div < min_kl_div) {
// 			return bo;
// 		}
// 		else {
// 			if (min_k == k) {
// 				return bo;
// 			}
// 			else {
// 				bo.setDiscretized(lvl, min_k);
// 			}
// 		}
// 	}
// 	else {
// 		if (min_k == k) {
// 			bo.setLastWord(-1);
// 			return bo;
// 		}
// 		else {
// 			bo.setDiscretized(lvl, min_k);
// 		}
// 	}
// 	return bo;
// }



/**
 * Compute the conditional probability of a word given a state.
 * This computation handles backoffs in the FST.
 */
real HierarchicalClusterFstBuilder::computeFstWordProb(VectorFst<LogArc> &fst, int word, FstIndex state) {
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
void HierarchicalClusterFstBuilder::computeOneBackoff(VectorFst<LogArc> &fst, FstIndex src, FstIndex bo) {
	real mass_normal = 0.0;
	real mass_bo = 0.0;
	
	//browse all non backoff edges from src
	//dprintf(1,"BACKOFF FOR NODE %li\n", src);
	MutableArcIterator<VectorFst<LogArc> > aiter(&fst, src);
	for (; !aiter.Done(); aiter.Next()) {
		const LogArc &src_arc = aiter.Value();
		if (src_arc.ilabel == EPSILON) { continue; } //skip epsilon edge
			//dprintf(2,"\tSRC:\tw%i\t%f\n", src_arc.ilabel, src_arc.weight.Value());
			//dprintf(2,"\tBCK:\tw%i\t%f\n", src_arc.ilabel, computeFstWordProb(fst, src_arc.ilabel, bo));
			mass_normal += myexp(src_arc.weight.Value());
			mass_bo += myexp(computeFstWordProb(fst, src_arc.ilabel, bo));
	}
	
	//dprintf(1,"Backoff = ((1 - %f)/(1 - %f)) = %f\n", mass_normal, mass_bo, ((1 - mass_normal)/(1 - mass_bo)));
	aiter.Reset(); //go back to the backoff arc, ie, the first arc.
	aiter.SetValue(LogArc(EPSILON,EPSILON, mylog(1 - mass_normal) - mylog(1 - mass_bo), bo));
	//dprintf(1,"\n\n");
}













/**
 * Compute the backoff weight for each backoff edge
 */
void HierarchicalClusterFstBuilder::computeAllBackoff(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred) {
	
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
			//dprintf(1,"\rBackoff = %i nodes / %i\t(%.6f %%)\t[ %.2f %% BO nodes proc'd ]", i, n_bo, (float) (100.0*i/n_bo), ((float) 100.0*(i/pred.size())) );
		}
	}
		printf("End of BO computation\n");
	cout << endl;
}












/**
 * Change destination of every edge arriving in "old_target" and replace with "new_target"
 */
void HierarchicalClusterFstBuilder::changeTargetForNode(VectorFst<LogArc> &fst,
                         FstIndex old_target,
                         FstIndex new_target) {
	for (StateIterator<VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
		StateId state = siter.Value();
		for (MutableArcIterator<VectorFst<LogArc> > aiter(&fst, state); !aiter.Done(); aiter.Next()) {
			const LogArc &arc = aiter.Value();
			if (arc.nextstate == old_target) {
				//dprintf(2,"\tMove %li --[%i / %f ]--> %li \tto\t%lu\n", state, arc.ilabel, arc.weight.Value(), old_target, new_target);
				 aiter.SetValue(LogArc(arc.ilabel, arc.olabel, arc.weight.Value(), new_target));
			}
		}
	}
}










/**
 * Remove a FST state ID "src" from the list of predecessors of FST state ID "dest"
 */
void HierarchicalClusterFstBuilder::removePred(map< FstIndex,vector<FstIndex> > &pred, FstIndex dest, FstIndex src) {
	vector<FstIndex>::iterator it;
	for (it = pred[dest].begin(); it != pred[dest].end(); ++it) {
		if (*it == src) {
			it = pred[dest].erase(it);
		}
	}
}









void HierarchicalClusterFstBuilder::removeStates(const VectorFst<LogArc> &old_fst, VectorFst<LogArc> &new_fst, vector<FstIndex> &to_be_deleted) {
	map<StateId,StateId> new_id;
	StateId shift = 0;
	sort(to_be_deleted.begin(), to_be_deleted.end());
	
	//compute new ids
	for (StateId i=0; i < old_fst.NumStates(); i++) {
		if (shift == to_be_deleted.size() || i != to_be_deleted[shift]) {
			//dprintf(1,"KEEP NODE %li\n",i);
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
				//dprintf(2,"NEW ARC: %li\t%i\t%f\t%li\n", new_id[state], arc.ilabel, exp(-arc.weight.Value()), new_id[arc.nextstate]);
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
vector<FstIndex> HierarchicalClusterFstBuilder::compactBackoffNodes(VectorFst<LogArc> &fst, map< FstIndex,set<FstIndex> > &pred, vector<bool> &non_bo_pred) {
	set<FstIndex>::iterator itv;
	set<FstIndex> v;
	map<FstIndex,FstIndex> redir;
	vector<FstIndex> deleted;
 	map< FstIndex,set<FstIndex> >::iterator itm;
 	int i = 0;
 	for(itm = pred.begin(); itm != pred.end(); ++itm) {
	 	fprintf(stderr, "\rCompaction - Step 1\t%i / %i\t(%.1f %%)", i, pred.size(), 100.0*((float) (i/pred.size())));
		MutableArcIterator<VectorFst<LogArc> > bo_node(&fst, itm->first);
		v = itm->second;
		for (itv = v.begin(); itv != v.end(); ++itv) {
		//if only one output (the backoff edge)
			if (fst.NumArcs(*itv) == 1) {
				//cout << "REDIR " << *itv << " TO " << itm->first << endl;
				if (*itv != itm->first) {
					redir[*itv] = itm->first;
				}
				//dprintf(2,"DIRECT REDIR %li TO %li\n", *itv, itm->first);
			}
		}
		i++;
	}
	

	FstIndex source;
	FstIndex target;
	map<FstIndex,FstIndex>::iterator transitivity;	
	i = 0;
	for (map<FstIndex,FstIndex>::iterator it_redir = redir.begin(); it_redir != redir.end(); ++it_redir) {
	 	fprintf(stderr, "\rCompaction - Step 2\t%i / %i\t(%.1f %%)", i, redir.size(), 100.0*((float) (i/redir.size())));
		source = it_redir->first;
		//try to find the final target, eg, target(a) = c whem a->b and b->c
		target = it_redir->second;
		transitivity = redir.find(target);
		while (transitivity != redir.end()) {
			target = transitivity->second;
			transitivity = redir.find(target);
		}
		if (target == source) { continue; }
		
		//dprintf(1,"REDIR %li TO %li\n", source, target);
		
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
		    	//dprintf(2,"\tAdd %li to bo list of %li\n", *it, target);
				pred[target].insert(*it);
			}
		}
		for (set<FstIndex>::iterator it = pred[target].begin();
		     it != pred[target].end();
		     ++it) {
		    if (*it == source) {
		    	//dprintf(2,"\tRemove %li from bo list of %li\n", source, target);
				pred[target].erase(it);
			}
		}
		pred.erase(source);
		
		i++;
		
	}
		printf("End of compaction\n");

//	fst.DeleteStates(deleted);
	return deleted;
	
}





real HierarchicalClusterFstBuilder::computeTotalEntropy(CRnnLM &rnnlm) {
	HierarchicalClusterDiscretizer *typed_dzer = dynamic_cast<HierarchicalClusterDiscretizer *>(dzer);

	HierarchicalClusterFstHistory fsth;
	int n_levels = typed_dzer->getNumLevels();
	int n_clusters = typed_dzer->getLevelSize(n_levels-1);
	
	real p_post = 1.0/(n_clusters*rnnlm.getVocabSize());
//	p_post = log(p_post);
	real adj_p_post;
	vector<real> p_cond(rnnlm.getVocabSize(), 0.0);
	real p = 0.0;
	real entropy = 0.0;
	real part_entropy = 0.0;

	for (int cid=0; cid < n_clusters; cid++) {
		fsth.setDiscretized(n_levels-1, cid); //set clusted id for the finest level
		for (int w=0; w < rnnlm.getVocabSize(); w++) {
			adj_p_post = p_post + 0.00001*(1.0-((float) w/ (float)rnnlm.getVocabSize()))
			                    -0.00001*((float) w/ (float)rnnlm.getVocabSize());
			fprintf(stderr, "\rCluster ID %i\tWord %i \tP_post = %f\tH=%.5f", cid, w, adj_p_post, entropy);
			adj_p_post = log(adj_p_post);
			fsth.setLastWord(w);
			computeEntropyAndConditionals(part_entropy, p_cond, rnnlm, fsth, adj_p_post);
			entropy += part_entropy;
		}
	}
	
	return pow(2.0,entropy/n_clusters);
	
}









/**
 * Compute all conditionals for a given discretize state (FstHistory)
 * and store them in a vector
 * Entropy is computed at the same time for speed reasons
 */
void HierarchicalClusterFstBuilder::computeEntropyAndConditionalsSpecial(real &entropy, vector<real> &res, CRnnLM &rnnlm, const FstHistory & fsth, real posterior) {
	struct neuron* output_layer = rnnlm.getOutputLayer();
	real p	 = 0.0;
	real p_joint = 0.0;
	int w=0;
	if (posterior > 0.0) { posterior = -posterior; }
	
	entropy = 0.0;
	
	fsth.loadAsInput(rnnlm, *dzer);
//	for (int i=0; i < 10; i++) {
//		rnnlm.copyHiddenLayerToInput();
//		rnnlm.computeNet(fsth.getLastWord(), 0);
//			printNeurons(rnnlm.getInputLayer(), 10000, 10010);
//			printNeurons(rnnlm.getHiddenLayer(), 0, 10);
//	}
	
	//store all conditionals
 	rnnlm.computeClassProbs(fsth.getLastWord());
	for (int c = 0; c < rnnlm.getClassSize(); c++) {
	 	rnnlm.computeClassWordProbs(fsth.getLastWord(), rnnlm.getWordFromClass(0, c));
		for (int i = 0; i < rnnlm.getNumWordsInClass(c); i++) {
			w = rnnlm.getWordFromClass(i, c);
			p = log(output_layer[rnnlm.getVocabSize()+c].ac)
			    +log(output_layer[w].ac);
			p_joint = posterior+p;
			entropy -= exp(p_joint)*p_joint;
			//compute and store P(w|current_state);
			res[w] = -p;	
		}
	}
}



// 
// real HierarchicalClusterFstBuilder::computeDeltaEntropy(real log_p_post, // -log
//                      real log_p_cond, // -log
//                      real log_p_cond_bo, // -log 
//                      real sum_seen, // real
//                      real sum_seen_bo) // real
// {
// 
// 	sum_seen -= 1e-10;
// 	sum_seen_bo -= 1e-10;
// 	log_p_post = -log_p_post;
// 	log_p_cond = -log_p_cond;
// 	log_p_cond_bo = -log_p_cond_bo;
// 	
// 	
// 	real log_old_bo = log(1.0 - sum_seen)
//                     - log(1.0 - sum_seen_bo);
// //	printf("bo(h) = 1.0 - %f / 1.0 - %f = %f\n", sum_seen, sum_seen_bo, (1.0 - sum_seen)/(1.0 - sum_seen_bo));
// 	real log_new_bo = log(1.0 - sum_seen + exp(log_p_cond))
//                     - log(1.0 - sum_seen_bo + exp(log_p_cond_bo));
// 	real removal = exp(log_p_cond)
// 	             * (log_p_cond_bo +log_new_bo -log_p_cond);
// 	real backed_off = (1.0-sum_seen)
// 	                * (log_new_bo *log_old_bo);
// 	                
// 	//dprintf(1,"P(h):\t\t%f\n", log_p_post);
// 	//dprintf(1,"P(w|h):\t\t%f\n", log_p_cond);
// 	//dprintf(1,"P'(w|h):\t%f\n", log_p_cond_bo + log_new_bo);
// 	//dprintf(1,"P(w|h'):\t%f\n", log_p_cond_bo);
// 	//dprintf(1,"sum_seen:\t%f\n", sum_seen);
// 	//dprintf(1,"sum'_seen:\t%f\n", sum_seen_bo);
// 	//dprintf(1,"bo(h):\t\t%f\n", exp(log_old_bo));
// 	//dprintf(1,"bo'(h):\t\t%f\n", exp(log_new_bo));
// 	//dprintf(1,"removal:\t%e\n", removal);
// 	//dprintf(1,"backed_off:\t%e\n", backed_off);
// 	//dprintf(1,"delta H:\t%e\n",  -exp(log_p_post)
// 	                        * (removal + backed_off));
// 	//dprintf(1,"delta PPL:\t%e\n",  exp(-exp(log_p_post)
// 	                        * (removal + backed_off)) -1.0);
// 
// 	return   -exp(log_p_post)
// 	       * (removal + backed_off);
// 
// }






real HierarchicalClusterFstBuilder::deltaProb(
               real p_cond,
               real p_cond_bo)
{
	p_cond = myexp(p_cond);
	p_cond_bo = myexp(p_cond_bo);
	//dprintf(1, "%e\t%e\t%f\n", p_cond, p_cond_bo,abs(p_cond-p_cond_bo)/p_cond);
	return abs(p_cond-p_cond_bo)/p_cond;
}




real HierarchicalClusterFstBuilder::computeDeltaEntropy(real log_p_post, // -log
                     real log_p_cond, // -log
                     real log_p_cond_bo, // -log 
                     real sum_seen, // real
                     real sum_seen_bo) // real
{

	sum_seen -= 1e-10;
	sum_seen_bo -= 1e-10;
	log_p_post = -log_p_post;
	log_p_cond = -log_p_cond;
	log_p_cond_bo = -log_p_cond_bo;
	
	
// 	real log_old_bo = log(1.0 - sum_seen)
//                     - log(1.0 - sum_seen_bo);
	real log_new_bo = log(1.0 - sum_seen + exp(log_p_cond))
                    - log(1.0 - sum_seen_bo + exp(log_p_cond_bo));
// 	real removal = exp(log_p_cond)
// 	             * (log_p_cond_bo +log_new_bo -log_p_cond);
// 	real backed_off = (1.0-sum_seen)
// 	                * (log_new_bo *log_old_bo);
	                
// 	//dprintf(1,"P(h):\t\t%f\n", log_p_post);
// 	//dprintf(1,"P(w|h):\t\t%f\n", log_p_cond);
// 	//dprintf(1,"P'(w|h):\t%f\n", log_p_cond_bo + log_new_bo);
// 	//dprintf(1,"P(w|h'):\t%f\n", log_p_cond_bo);
// 	//dprintf(1,"sum_seen:\t%f\n", sum_seen);
// 	//dprintf(1,"sum'_seen:\t%f\n", sum_seen_bo);
// 	//dprintf(1,"bo(h):\t\t%f\n", exp(log_old_bo));
// 	//dprintf(1,"bo'(h):\t\t%f\n", exp(log_new_bo));
// 	//dprintf(1,"removal:\t%e\n", removal);
// 	//dprintf(1,"backed_off:\t%e\n", backed_off);
// 	//dprintf(1,"delta H:\t%e\n",  -exp(log_p_post)
// 	                        * (removal + backed_off));
// 	//dprintf(1,"delta PPL:\t%e\n",  exp(-exp(log_p_post)
// 	                        * (removal + backed_off)) -1.0);
// 	//dprintf(1,"ratio P(w|.):\t%e\n", abs(exp(log_p_cond_bo + log_new_bo)-exp(log_p_cond))/exp(log_p_cond));
real p = log_p_cond+log_p_post;
real a = exp(log_p_cond_bo + log_new_bo);
real b = exp(log_p_cond);
real c = -p*exp(p);
	                    
 	//dprintf(1,"ratio P(w|.):\t%e\n", c*abs(a-b)/b);    
	return c*(abs(a-b)/b);
//  	return abs(a-b)*exp(log_p_post);
//	return abs(exp(log_p_cond_bo + log_new_bo)-exp(log_p_cond))/exp(log_p_cond);

// 	return   -exp(log_p_post)
// 	       * (removal + backed_off);

}




void HierarchicalClusterFstBuilder::estimateMasses(real *mass1, //out
                   real *mass2, //out
                   CRnnLM & rnnlm,
                   real threshold, //in
                   real p_post, //in
                   vector<real> &all_prob, //in
                   vector<real> &all_bo_prob //in
                   ) {

	real p, delta;
	*mass1 = 0.5;
	*mass2 = 0.5;
	debug_mode--;
	for (int i=0; i<4; i++) {
		real new_mass1 = 1.0;
		real new_mass2 = 1.0;
// 		printf("Old mass1 = %.2f\n",*mass1);
// 		printf("Old mass2 = %.2f\n",*mass2);
		for (int w=0; w < rnnlm.getVocabSize(); w++) {
			p = all_prob[w];
			delta = exp(computeDeltaEntropy(p_post,
			                                p,
			                                all_bo_prob[w],
			                                *mass1,
			                                *mass2)) -1.0;
			if (delta <= threshold) {
// 				printf("\t\tPrune\n");
				new_mass1 -= exp(-p);
				new_mass2 -= exp(-all_bo_prob[w]);
			}

		}
// 		printf("New mass1 = %.2f\n",new_mass1);
// 		printf("New mass2 = %.2f\n",new_mass2);
		if (abs(*mass1-new_mass1) < 0.1) {
			*mass1 = (new_mass1+*mass1)/2.0;
			*mass2 = (new_mass2+*mass2)/2.0;
			break;
		}
// 		else if (*mass1 <= new_mass1) {
// 			*mass1 = new_mass1;
// 			*mass2 = new_mass2;
// 			break;
// 		}
		else {
			*mass1 = (new_mass1+*mass1)/2.0;
			*mass2 = (new_mass2+*mass2)/2.0;
		}
	}
	debug_mode++;
// 	printf("Final mass1 = %.2f\n",*mass1);
// 	printf("Final mass2 = %.2f\n",*mass2);
// 	printf("--\n");
	return;	
}





/* ========================================================================================================
                                            CONVERTION METHOD
   ======================================================================================================== */







/**
 * Create an FST based on an RNN
 */
void HierarchicalClusterFstBuilder::convertRNN(CRnnLM & rnnlm, VectorFst<LogArc> &fst) {
	queue<HierarchicalClusterFstHistory> q;
	VectorFst<LogArc> new_fst;
	HierarchicalClusterDiscretizer *typed_dzer = dynamic_cast<HierarchicalClusterDiscretizer *>(dzer);
	
	HierarchicalClusterFstHistory fsth;
	FstIndex id = 0;
	
	HierarchicalClusterFstHistory new_fsth;
	FstIndex new_id;

	HierarchicalClusterFstHistory min_backoff;
	set<HierarchicalClusterFstHistory>set_min_backoff;
	
	HierarchicalClusterFstHistory bo_fsth;
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
//	printHierarchicalClusters(rnnlm.getInputLayer(),0,10);

	// Initial state ( 0 | hidden layer after </s>)
	fsth.setFstHistory(rnnlm, *dzer);
	fsth.setLastWord(0);
	q.push(fsth);
	addFstState(id, new HierarchicalClusterFstHistory(fsth), fst);
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
		state2h.push_back(new HierarchicalClusterFstHistory(fsth));


		if (id == FINAL_STATE) { continue; }
		

 		int disc_lvl = fsth.getNumClusters()-1;
//Original backoff place
 		bo_fsth = getBackoff(rnnlm, fsth, set_min_backoff, all_prob, to_be_removed);
//   		bo_fsth = getBackoff(rnnlm, fsth, *typed_dzer, all_prob, to_be_removed);
/*************************** BACKOFF LOCAL ENTROPY ***********************************/  		
		if (fsth.getLastWord() > -1) {
			bo_post = typed_dzer->getPrior(bo_fsth.getNumClusters()-1, bo_fsth.getFinestDiscretized())
			        + mylog((float) rnnlm.getWordCount(bo_fsth.getLastWord())/total_counts);
//			        + mylog(1.0/rnnlm.getVocabSize());
		}
		else {
			bo_post = 0.0;
		}
		computeEntropyAndConditionals(bo_entropy,
		                              all_bo_prob,
		                              rnnlm,
		                              bo_fsth,
		                              bo_post);
					      

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
//	deleted = compactBackoffNodes(fst, pred, non_bo_pred);
	computeAllBackoff(fst, pred);


	//remove useless nodes
//	removeStates(fst, new_fst, deleted);
//	fst.DeleteStates();
//	fst = new_fst;
	
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





