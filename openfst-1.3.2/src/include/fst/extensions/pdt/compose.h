// compose.h

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: riley@google.com (Michael Riley)
//
// \file
// Compose a PDT and an FST.

#ifndef FST_EXTENSIONS_PDT_COMPOSE_H__
#define FST_EXTENSIONS_PDT_COMPOSE_H__

#include <fst/compose.h>

namespace fst {

// Class to setup composition options for PDT composition.
// Default is for the PDT as the first composition argument.
template <class Arc, bool left_pdt = true>
class PdtComposeOptions : public
ComposeFstOptions<Arc,
                  MultiEpsMatcher< Matcher<Fst<Arc> > >,
                  MultiEpsFilter<AltSequenceComposeFilter<
                                   MultiEpsMatcher<
                                     Matcher<Fst<Arc> > > > > > {
 public:
  typedef typename Arc::Label Label;
  typedef MultiEpsMatcher< Matcher<Fst<Arc> > > PdtMatcher;
  typedef MultiEpsFilter<AltSequenceComposeFilter<PdtMatcher> > PdtFilter;
  typedef ComposeFstOptions<Arc, PdtMatcher, PdtFilter> COptions;
  using COptions::matcher1;
  using COptions::matcher2;
  using COptions::filter;

  PdtComposeOptions(const Fst<Arc> &ifst1,
                    const vector<pair<Label, Label> > &parens,
                    const Fst<Arc> &ifst2) {
    matcher1 = new PdtMatcher(ifst1, MATCH_OUTPUT, kMultiEpsList);
    matcher2 = new PdtMatcher(ifst2, MATCH_INPUT, kMultiEpsLoop);

    // Treat parens as multi-epsilons when composing.
    for (size_t i = 0; i < parens.size(); ++i) {
      matcher1->AddMultiEpsLabel(parens[i].first);
      matcher1->AddMultiEpsLabel(parens[i].second);
      matcher2->AddMultiEpsLabel(parens[i].first);
      matcher2->AddMultiEpsLabel(parens[i].second);
    }

    filter = new PdtFilter(ifst1, ifst2, matcher1, matcher2, true);
  }
};

// Class to setup composition options for PDT with FST composition.
// Specialization is for the FST as the first composition argument.
template <class Arc>
class PdtComposeOptions<Arc, false> : public
ComposeFstOptions<Arc,
                  MultiEpsMatcher< Matcher<Fst<Arc> > >,
                  MultiEpsFilter<SequenceComposeFilter<
                                   MultiEpsMatcher<
                                     Matcher<Fst<Arc> > > > > > {
 public:
  typedef typename Arc::Label Label;
  typedef MultiEpsMatcher< Matcher<Fst<Arc> > > PdtMatcher;
  typedef MultiEpsFilter<SequenceComposeFilter<PdtMatcher> > PdtFilter;
  typedef ComposeFstOptions<Arc, PdtMatcher, PdtFilter> COptions;
  using COptions::matcher1;
  using COptions::matcher2;
  using COptions::filter;

  PdtComposeOptions(const Fst<Arc> &ifst1,
                    const Fst<Arc> &ifst2,
                    const vector<pair<Label, Label> > &parens) {
    matcher1 = new PdtMatcher(ifst1, MATCH_OUTPUT, kMultiEpsLoop);
    matcher2 = new PdtMatcher(ifst2, MATCH_INPUT, kMultiEpsList);

    // Treat parens as multi-epsilons when composing.
    for (size_t i = 0; i < parens.size(); ++i) {
      matcher1->AddMultiEpsLabel(parens[i].first);
      matcher1->AddMultiEpsLabel(parens[i].second);
      matcher2->AddMultiEpsLabel(parens[i].first);
      matcher2->AddMultiEpsLabel(parens[i].second);
    }

    filter = new PdtFilter(ifst1, ifst2, matcher1, matcher2, true);
  }
};


// Composes pushdown transducer (PDT) encoded as an FST (1st arg) and
// an FST (2nd arg) with the result also a PDT encoded as an Fst. (3rd arg).
// In the PDTs, some transitions are labeled with open or close
// parentheses. To be interpreted as a PDT, the parens must balance on
// a path (see PdtExpand()). The open-close parenthesis label pairs
// are passed in 'parens'.
template <class Arc>
void Compose(const Fst<Arc> &ifst1,
             const vector<pair<typename Arc::Label,
                               typename Arc::Label> > &parens,
             const Fst<Arc> &ifst2,
             MutableFst<Arc> *ofst,
             const ComposeOptions &opts = ComposeOptions()) {

  PdtComposeOptions<Arc, true> copts(ifst1, parens, ifst2);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect)
    Connect(ofst);
}


// Composes an FST (1st arg) and pushdown transducer (PDT) encoded as
// an FST (2nd arg) with the result also a PDT encoded as an Fst (3rd arg).
// In the PDTs, some transitions are labeled with open or close
// parentheses. To be interpreted as a PDT, the parens must balance on
// a path (see ExpandFst()). The open-close parenthesis label pairs
// are passed in 'parens'.
template <class Arc>
void Compose(const Fst<Arc> &ifst1,
             const Fst<Arc> &ifst2,
             const vector<pair<typename Arc::Label,
                               typename Arc::Label> > &parens,
             MutableFst<Arc> *ofst,
             const ComposeOptions &opts = ComposeOptions()) {

  PdtComposeOptions<Arc, false> copts(ifst1, ifst2, parens);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect)
    Connect(ofst);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_COMPOSE_H__
