// bi-table.h

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
// Classes for representing a bijective mapping between an arbitrary entry
// of type T and a signed integral ID.

#ifndef FST_LIB_BI_TABLE_H__
#define FST_LIB_BI_TABLE_H__

#include <deque>
#include <vector>
using std::vector;

namespace fst {

// BI TABLES - these determine a bijective mapping between an
// arbitrary entry of type T and an signed integral ID of type I. The IDs are
// allocated starting from 0 in order.
//
// template <class I, class T>
// class BiTable {
//  public:
//
//   // Required constructors.
//   BiTable();
//
//   // Lookup integer ID from entry. If it doesn't exist and 'insert'
//   / is true, then add it. Otherwise return -1.
//   I FindId(const T &entry, bool insert = true);
//   // Lookup entry from integer ID.
//   const T &FindEntry(I) const;
//   // # of stored entries.
//   I Size() const;
// };

// An implementation using a hash map for the entry to ID mapping.
// The  entry T must have == defined and the default constructor
// must produce an entry that will never be seen. H is the hash function.
template <class I, class T, class H>
class HashBiTable {
 public:

  HashBiTable() {
    T empty_entry;
  }

  I FindId(const T &entry, bool insert = true) {
    I &id_ref = entry2id_[entry];
    if (id_ref == 0) {  // T not found
      if (insert) {     // store and assign it a new ID
        id2entry_.push_back(entry);
        id_ref = id2entry_.size();
      } else {
        return -1;
      }
    }
    return id_ref - 1;  // NB: id_ref = ID + 1
  }

  const T &FindEntry(I s) const {
    return id2entry_[s];
  }

  I Size() const { return id2entry_.size(); }

 private:
  unordered_map<T, I, H> entry2id_;
  vector<T> id2entry_;

  DISALLOW_COPY_AND_ASSIGN(HashBiTable);
};


// An implementation using a hash set for the entry to ID
// mapping.  The hash set holds 'keys' which are either the ID
// or kCurrentKey.  These keys can be mapped to entrys either by
// looking up in the entry vector or, if kCurrentKey, in current_entry_
// member. The hash and key equality functions map to entries first.
// The  entry T must have == defined and the default constructor
// must produce a entry that will never be seen. H is the hash
// function.
template <class I, class T, class H>
class CompactHashBiTable {
 public:
  friend class HashFunc;
  friend class HashEqual;

  CompactHashBiTable()
      : hash_func_(*this),
        hash_equal_(*this),
        keys_(0, hash_func_, hash_equal_) {
  }

  // Reserves space for table_size elements.
  explicit CompactHashBiTable(size_t table_size)
      : hash_func_(*this),
        hash_equal_(*this),
        keys_(table_size, hash_func_, hash_equal_) {
    id2entry_.reserve(table_size);
  }

  I FindId(const T &entry, bool insert = true) {
    current_entry_ = &entry;
    typename KeyHashSet::const_iterator it = keys_.find(kCurrentKey);
    if (it == keys_.end()) {  // T not found
      if (insert) {           // store and assign it a new ID
        I key = id2entry_.size();
        id2entry_.push_back(entry);
        keys_.insert(key);
        return key;
      } else {
        return -1;
      }
    } else {
      return *it;
    }
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }
  I Size() const { return id2entry_.size(); }

 private:
  static const I kEmptyKey;    // -1
  static const I kCurrentKey;  // -2

  class HashFunc {
   public:
    HashFunc(const CompactHashBiTable &ht) : ht_(&ht) {}

    size_t operator()(I k) const { return hf(ht_->Key2T(k)); }
   private:
    const CompactHashBiTable *ht_;
    H hf;
  };

  class HashEqual {
   public:
    HashEqual(const CompactHashBiTable &ht) : ht_(&ht) {}

    bool operator()(I k1, I k2) const {
      return ht_->Key2T(k1) == ht_->Key2T(k2);
    }
   private:
    const CompactHashBiTable *ht_;
  };

  typedef unordered_set<I, HashFunc, HashEqual> KeyHashSet;

  const T &Key2T(I k) const {
    if (k == kEmptyKey)
      return empty_entry_;
    else if (k == kCurrentKey)
      return *current_entry_;
    else
      return id2entry_[k];
  }

  HashFunc hash_func_;
  HashEqual hash_equal_;
  KeyHashSet keys_;
  vector<T> id2entry_;
  const T empty_entry_;
  const T *current_entry_;

  DISALLOW_COPY_AND_ASSIGN(CompactHashBiTable);
};

template <class I, class T, class H>
const I CompactHashBiTable<I, T, H>::kEmptyKey = -1;

template <class I, class T, class H>
const I CompactHashBiTable<I, T, H>::kCurrentKey = -2;


// An implementation using a vector for the entry to ID mapping.
// It is passed a function object FP that should fingerprint entries
// uniquely to an integer that can used as a vector index. Normally,
// VectorBiTable constructs the FP object.  The user can instead
// pass in this object; in that case, VectorBiTable takes its
// ownership.
template <class I, class T, class FP>
class VectorBiTable {
 public:
  explicit VectorBiTable(FP *fp = 0) : fp_(fp ? fp : new FP()) {}

  ~VectorBiTable() { delete fp_; }

  I FindId(const T &entry, bool insert = true) {
    ssize_t fp = (*fp_)(entry);
    if (fp >= fp2id_.size())
      fp2id_.resize(fp + 1);
    I &id_ref = fp2id_[fp];
    if (id_ref == 0) {  // T not found
      if (insert) {     // store and assign it a new ID
        id2entry_.push_back(entry);
        id_ref = id2entry_.size();
      } else {
        return -1;
      }
    }
    return id_ref - 1;  // NB: id_ref = ID + 1
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }

  I Size() const { return id2entry_.size(); }

  const FP &Fingerprint() const { return *fp_; }

 private:
  FP *fp_;
  vector<I> fp2id_;
  vector<T> id2entry_;

  DISALLOW_COPY_AND_ASSIGN(VectorBiTable);
};


// An implementation using a vector and a compact hash table. The
// selecting functor S returns true for entries to be hashed in the
// vector.  The fingerprinting functor FP returns a unique fingerprint
// for each entry to be hashed in the vector (these need to be
// suitable for indexing in a vector).  The hash functor H is used when
// hashing entry into the compact hash table.
template <class I, class T, class S, class FP, class H>
class VectorHashBiTable {
 public:
  friend class HashFunc;
  friend class HashEqual;

  VectorHashBiTable(S *s, FP *fp, H *h,
                       size_t vector_size = 0,
                       size_t entry_size = 0)
      : selector_(s),
        fp_(fp),
        h_(h),
        hash_func_(*this),
        hash_equal_(*this),
        keys_(0, hash_func_, hash_equal_) {
    if (vector_size)
      fp2id_.reserve(vector_size);
    if (entry_size)
      id2entry_.reserve(entry_size);
  }

  ~VectorHashBiTable() {
    delete selector_;
    delete fp_;
    delete h_;
  }

  I FindId(const T &entry, bool insert = true) {
    if ((*selector_)(entry)) {  // Use the vector if 'selector_(entry) == true'
      uint64 fp = (*fp_)(entry);
      if (fp2id_.size() <= fp)
        fp2id_.resize(fp + 1, 0);
      if (fp2id_[fp] == 0) {         // T not found
        if (insert) {                // store and assign it a new ID
          id2entry_.push_back(entry);
          fp2id_[fp] = id2entry_.size();
        } else {
          return -1;
        }
      }
      return fp2id_[fp] - 1;  // NB: assoc_value = ID + 1
    } else {  // Use the hash table otherwise.
      current_entry_ = &entry;
      typename KeyHashSet::const_iterator it = keys_.find(kCurrentKey);
      if (it == keys_.end()) {
        if (insert) {
          I key = id2entry_.size();
          id2entry_.push_back(entry);
          keys_.insert(key);
          return key;
        } else {
          return -1;
        }
      } else {
        return *it;
      }
    }
  }

  const T &FindEntry(I s) const {
    return id2entry_[s];
  }

  I Size() const { return id2entry_.size(); }

  const S &Selector() const { return *selector_; }

  const FP &Fingerprint() const { return *fp_; }

  const H &Hash() const { return *h_; }

 private:
  static const I kEmptyKey;
  static const I kCurrentKey;

  class HashFunc {
   public:
    HashFunc(const VectorHashBiTable &ht) : ht_(&ht) {}

    size_t operator()(I k) const { return (*(ht_->h_))(ht_->Key2Entry(k)); }
   private:
    const VectorHashBiTable *ht_;
  };

  class HashEqual {
   public:
    HashEqual(const VectorHashBiTable &ht) : ht_(&ht) {}

    bool operator()(I k1, I k2) const {
      return ht_->Key2Entry(k1) == ht_->Key2Entry(k2);
    }
   private:
    const VectorHashBiTable *ht_;
  };

  typedef unordered_set<I, HashFunc, HashEqual> KeyHashSet;

  const T &Key2Entry(I k) const {
    if (k == kEmptyKey)
      return empty_entry_;
    else if (k == kCurrentKey)
      return *current_entry_;
    else
      return id2entry_[k];
  }


  S *selector_;  // Returns true if entry hashed into vector
  FP *fp_;       // Fingerprint used when hashing entry into vector
  H *h_;         // Hash function used when hashing entry into hash_set

  vector<T> id2entry_;  // Maps state IDs to entry
  vector<I> fp2id_;        // Maps entry fingerprints to IDs

  // Compact implementation of the hash table mapping entrys to
  // state IDs using the hash function 'h_'
  HashFunc hash_func_;
  HashEqual hash_equal_;
  KeyHashSet keys_;
  const T empty_entry_;
  const T *current_entry_;

  DISALLOW_COPY_AND_ASSIGN(VectorHashBiTable);
};

template <class I, class T, class S, class FP, class H>
const I VectorHashBiTable<I, T, S, FP, H>::kEmptyKey = -1;

template <class I, class T, class S, class FP, class H>
const I VectorHashBiTable<I, T, S, FP, H>::kCurrentKey = -2;


// An implementation using a hash map for the entry to ID
// mapping. This version permits erasing of s.  The  entry T
// must have == defined and its default constructor must produce a
// entry that will never be seen. F is the hash function.
template <class I, class T, class F>
class ErasableBiTable {
 public:
  ErasableBiTable() : first_(0) {}

  I FindId(const T &entry, bool insert = true) {
    I &id_ref = entry2id_[entry];
    if (id_ref == 0) {  // T not found
      if (insert) {     // store and assign it a new ID
        id2entry_.push_back(entry);
        id_ref = id2entry_.size() + first_;
      } else {
        return -1;
      }
    }
    return id_ref - 1;  // NB: id_ref = ID + 1
  }

  const T &FindEntry(I s) const { return id2entry_[s - first_]; }

  I Size() const { return id2entry_.size(); }

  void Erase(I s) {
    T &entry = id2entry_[s - first_];
    typename unordered_map<T, I, F>::iterator it =
        entry2id_.find(entry);
    entry2id_.erase(it);
    id2entry_[s - first_] = empty_entry_;
    while (!id2entry_.empty() && id2entry_.front() == empty_entry_) {
      id2entry_.pop_front();
      ++first_;
    }
  }

 private:
  unordered_map<T, I, F> entry2id_;
  deque<T> id2entry_;
  const T empty_entry_;
  I first_;        // I of first element in the deque;

  DISALLOW_COPY_AND_ASSIGN(ErasableBiTable);
};

}  // namespace fst

#endif  // FST_LIB_BI_TABLE_H__
