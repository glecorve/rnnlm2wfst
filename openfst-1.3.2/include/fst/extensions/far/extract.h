// extract-main.h

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
// Modified: jpr@google.com (Jake Ratkiewicz) to use the new arc-dispatch

// \file
// Extracts component FSTs from an finite-state archive.
//

#ifndef FST_EXTENSIONS_FAR_EXTRACT_H__
#define FST_EXTENSIONS_FAR_EXTRACT_H__

#include <string>
#include <vector>
using std::vector;

#include <fst/extensions/far/far.h>

namespace fst {

template<class Arc>
void FarExtract(const vector<string> &ifilenames,
                const int32 &generate_filenames,
                const string &begin_key,
                const string &end_key,
                const string &filename_prefix,
                const string &filename_suffix) {
  FarReader<Arc> *far_reader = FarReader<Arc>::Open(ifilenames);
  if (!far_reader) return;

  if (!begin_key.empty())
    far_reader->Find(begin_key);

  string okey;
  int nrep = 0;
  for (int i = 1; !far_reader->Done(); far_reader->Next(), ++i) {
    string key = far_reader->GetKey();
    if (!end_key.empty() && end_key < key)
      break;
    const Fst<Arc> &fst = far_reader->GetFst();

    if (key == okey)
      ++nrep;
    else
      nrep = 0;

    okey = key;

    string ofilename;
    if (generate_filenames) {
      ostringstream tmp;
      tmp.width(generate_filenames);
      tmp.fill('0');
      tmp << i;
      ofilename = tmp.str();
    } else {
      if (nrep > 0) {
        ostringstream tmp;
        tmp << '.' << nrep;
        key.append(tmp.str().data(), tmp.str().size());
      }
      ofilename = key;
    }
    fst.Write(filename_prefix + ofilename + filename_suffix);
  }

  return;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_EXTRACT_H__
