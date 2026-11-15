// Copyright 2015 The etcd Authors
//
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

package mvcc

import (
	"bytes"
	"errors"
	"sync"

	"go.uber.org/zap"
	"k8s.io/utils/third_party/forked/golang/btree"
)

var ErrRevisionNotFound = errors.New("mvcc: revision not found")

type index interface {
	Get(key []byte, atRev int64) (rev, created Revision, ver int64, err error)
	Range(key, end []byte, atRev int64, limit int, withTotalCount bool) (keys [][]byte, modifies, creates []Revision, versions []int64, totalCount int)
	Revisions(key, end []byte, atRev int64, limit int, withTotalCount bool) ([]Revision, int)
	CountRevisions(key, end []byte, atRev int64) int
	Put(key []byte, rev Revision)
	Tombstone(key []byte, rev Revision) error
	// RestoreCompactRevision, Restore and RestoreTombstone restore an empty
	// index from the backend. The backend has lost revisions before its
	// compacted revision, and before the revision of an interrupted
	// compaction, so RestoreCompactRevision makes the index start at that
	// revision. Restore and RestoreTombstone then add the revisions in the
	// backend, in order. What is left of the revisions before the compacted
	// one is the state at the compacted one, and not a history.
	RestoreCompactRevision(rev int64)
	Restore(key []byte, rev, created Revision, ver int64)
	// RestoreTombstone accepts a key missing from the index. Compaction may keep
	// a tombstone at the compact revision while removing all earlier revisions
	// of the key. Restore the tombstone into the index so that resumed compaction
	// keeps it in the backend for watches starting at that revision.
	//
	// See https://github.com/etcd-io/etcd/issues/19179
	// and https://github.com/etcd-io/etcd/pull/19188.
	RestoreTombstone(key []byte, rev Revision)
	Compact(rev int64) map[Revision]struct{}
	Keep(rev int64) map[Revision]struct{}
}

// treeIndex keeps a B-tree of the keys for each main revision since the last
// compaction, see etcd-io/etcd#18184. A revision clones the previous tree and
// copies only the paths that it changes.
//
// Like the store, it starts at revision 1 without keys, and every write is at
// the latest revision or at the one after it. So it has every revision from
// the compacted one to the latest one. A read before the compacted revision
// gets ErrCompacted or no keys. Anything else at a revision that the index
// does not have panics, because the store does not do that.
//
// For example, after put a at 2, put b at 3 and delete a at 4:
//
//	snapshot     [2]    [3]      [4]
//	tree         {a}    {a, b}   {b}
//	tombstones                   a@4
//
// and the trees of two revisions share the nodes that the later one did not
// change:
//
//	[3]  root          [4]  root'
//	     /  \               /  \
//	    A    B             A'   B      A' is a copy of A without a, B is shared
//
// Readers get their tree under the read lock and walk it without a lock. A
// transaction changes the tree of its revision in place, which is safe because
// the store does not read a revision before its transaction ends.
type treeIndex struct {
	sync.RWMutex
	snapshots snapshots // never empty
	lg        *zap.Logger
}

// keyRev is the latest revision of a key. Trees share it, so it must not
// change after the write that inserts it.
type keyRev struct {
	key          []byte
	mod, created Revision
	version      int64
}

// treeDegree is 16, not 32 like the per-key tree it replaces: a revision
// copies a node per level, and 16 was smaller and faster in the benchmarks.
const treeDegree = 16

func keyRevLess(a, b *keyRev) bool {
	return bytes.Compare(a.key, b.key) < 0
}

func newTreeIndex(lg *zap.Logger) index {
	ti := &treeIndex{lg: lg}
	// The revisions 0 and 1 have no keys. The first write of the store is at
	// revision 2, and the store can be compacted at 0 or 1 before it.
	tree := btree.New(treeDegree, keyRevLess)
	ti.snapshots.Append(snapshot{rev: 0, tree: tree})
	ti.snapshots.Append(snapshot{rev: 1, tree: tree.Clone()})
	return ti
}

func (ti *treeIndex) Put(key []byte, rev Revision) {
	ti.Lock()
	defer ti.Unlock()
	kr := &keyRev{key: key, mod: rev, created: rev, version: 1}
	// Insert first, so that the tree is walked once. Nothing reads kr before
	// the transaction of rev ends.
	if prev := insert(ti.unsafeForWrite(rev.Main), kr); prev != nil {
		kr.created, kr.version = prev.created, prev.version+1
	}
}

func (ti *treeIndex) Tombstone(key []byte, rev Revision) error {
	ti.Lock()
	defer ti.Unlock()
	probe := &keyRev{key: key}
	if _, ok := ti.snapshots.Latest().tree.Get(probe); !ok {
		return ErrRevisionNotFound
	}
	sn := ti.unsafeForWrite(rev.Main)
	sn.tree.Delete(probe)
	sn.tombstones = append(sn.tombstones, rev)
	keysGauge.Dec()
	return nil
}

// RestoreCompactRevision makes the index, which must be empty, start at rev.
func (ti *treeIndex) RestoreCompactRevision(rev int64) {
	ti.Lock()
	defer ti.Unlock()
	latest := ti.snapshots.Latest()
	if latest.rev != 1 || latest.tree.Len() != 0 {
		ti.lg.Panic("the index is not empty when it is restored",
			zap.Int64("latest-revision", latest.rev), zap.Int64("compact-revision", rev))
	}
	ti.snapshots = snapshots{}
	ti.snapshots.Append(snapshot{rev: rev, tree: latest.tree})
}

// Restore takes created and ver from the backend, because compaction may have
// removed the earlier revisions of the key.
func (ti *treeIndex) Restore(key []byte, rev, created Revision, ver int64) {
	ti.Lock()
	defer ti.Unlock()
	insert(ti.unsafeForRestore(rev.Main), &keyRev{key: key, mod: rev, created: created, version: ver})
}

// RestoreTombstone accepts a missing key, because compaction keeps a tombstone
// but removes the revisions before it.
func (ti *treeIndex) RestoreTombstone(key []byte, rev Revision) {
	ti.Lock()
	defer ti.Unlock()
	sn := ti.unsafeForRestore(rev.Main)
	if _, ok := sn.tree.Delete(&keyRev{key: key}); ok {
		keysGauge.Dec()
	}
	// A tombstone folded into the compacted state is not a tombstone at it.
	if rev.Main == sn.rev {
		sn.tombstones = append(sn.tombstones, rev)
	}
}

// insert adds kr to sn and returns the revision that it replaces, if any. kr
// keeps the key bytes of that revision rather than a copy per revision.
func insert(sn *snapshot, kr *keyRev) *keyRev {
	prev, ok := sn.tree.ReplaceOrInsert(kr)
	if !ok {
		keysGauge.Inc()
		return nil
	}
	kr.key = prev.key
	return prev
}

// unsafeForWrite returns the snapshot that writes at main change: the latest
// one, or a new one if main is the revision after it. The write lock must be
// held. It panics at any other revision.
func (ti *treeIndex) unsafeForWrite(main int64) *snapshot {
	latest := ti.snapshots.Latest()
	switch main {
	case latest.rev:
		return latest
	case latest.rev + 1:
		return ti.snapshots.Append(snapshot{rev: main, tree: latest.tree.Clone()})
	}
	ti.lg.Panic("revision is not the latest revision in the index or the one after it",
		zap.Int64("latest-revision", latest.rev), zap.Int64("revision", main))
	return nil
}

// unsafeForRestore is unsafeForWrite, except that revisions before the
// compacted one go into the compacted state while that is the only snapshot.
// They are what the compaction kept, and what an interrupted compaction did
// not delete yet.
//
// For example, with the index compacted at 5, and the revisions 3, 5, 7 and 9
// in the backend:
//
//	backend     3    5    7    9
//	             \  /     |    |
//	snapshot     [5]     [7]  [9]
func (ti *treeIndex) unsafeForRestore(main int64) *snapshot {
	if latest := ti.snapshots.Latest(); ti.snapshots.Len() == 1 && main < latest.rev {
		return latest
	}
	return ti.unsafeForWrite(main)
}

// Get returns ErrCompacted if atRev was compacted.
func (ti *treeIndex) Get(key []byte, atRev int64) (modified, created Revision, ver int64, err error) {
	sn, ok := ti.at(atRev)
	if !ok {
		return Revision{}, Revision{}, 0, ErrCompacted
	}
	kr, ok := sn.tree.Get(&keyRev{key: key})
	if !ok {
		return Revision{}, Revision{}, 0, ErrRevisionNotFound
	}
	return kr.mod, kr.created, kr.version, nil
}

// Revisions returns limited number of revisions from key(included) to end(excluded)
// at the given rev. The returned slice is sorted in the order of key. There is no limit if limit <= 0.
// The second return parameter isn't capped by the limit and reflects the total number of revisions.
func (ti *treeIndex) Revisions(key, end []byte, atRev int64, limit int, withTotalCount bool) (revs []Revision, total int) {
	total = ti.visit(key, end, atRev, limit, withTotalCount, func(kr *keyRev) {
		revs = append(revs, kr.mod)
	})
	return revs, total
}

// CountRevisions returns the number of revisions
// from key(included) to end(excluded) at the given rev.
func (ti *treeIndex) CountRevisions(key, end []byte, atRev int64) int {
	return ti.visit(key, end, atRev, 0, false, func(*keyRev) {})
}

func (ti *treeIndex) Range(key, end []byte, atRev int64, limit int, withTotalCount bool) (keys [][]byte, modifies, creates []Revision, versions []int64, total int) {
	total = ti.visit(key, end, atRev, limit, withTotalCount, func(kr *keyRev) {
		keys = append(keys, kr.key)
		modifies = append(modifies, kr.mod)
		creates = append(creates, kr.created)
		versions = append(versions, kr.version)
	})
	return keys, modifies, creates, versions, total
}

// visit calls f in key order for the keys in the range at atRev, and returns
// their number. The range is
//
//	end == nil      key only
//	len(end) == 0   [key, the last key]
//	otherwise       [key, end)
//
// With limit > 0, f gets the first limit keys, and the count stops there too
// unless withTotalCount is set. A compacted atRev has no keys.
func (ti *treeIndex) visit(key, end []byte, atRev int64, limit int, withTotalCount bool, f func(*keyRev)) (total int) {
	sn, ok := ti.at(atRev)
	if !ok {
		return 0
	}
	if end == nil {
		if kr, ok := sn.tree.Get(&keyRev{key: key}); ok {
			f(kr)
			return 1
		}
		return 0
	}
	each := func(kr *keyRev) bool {
		if limit <= 0 || total < limit {
			f(kr)
		} else if !withTotalCount {
			return false
		}
		total++
		return true
	}
	if len(end) == 0 {
		sn.tree.AscendGreaterOrEqual(&keyRev{key: key}, each)
	} else {
		sn.tree.AscendRange(&keyRev{key: key}, &keyRev{key: end}, each)
	}
	return total
}

// Compact drops the snapshots before rev, and returns the revisions that the
// backend keeps: the latest revision of each key at rev, and the tombstones at
// rev, which a watch from rev needs, see etcd-io/etcd#18089.
func (ti *treeIndex) Compact(rev int64) map[Revision]struct{} {
	ti.lg.Info("compact tree index", zap.Int64("revision", rev))
	sn := ti.mustAt(rev)
	keep := latestRevisions(sn.tree)
	for _, r := range sn.tombstones {
		keep[r] = struct{}{}
	}

	ti.Lock()
	defer ti.Unlock()
	ti.snapshots.Compact(rev)
	return keep
}

// Keep is Compact without the tombstones at rev, which v3.5.15, v3.4.33 and
// older versions delete, so that the hash is the same on every version.
func (ti *treeIndex) Keep(rev int64) map[Revision]struct{} {
	return latestRevisions(ti.mustAt(rev).tree)
}

// latestRevisions returns the latest revision of each key in tree.
func latestRevisions(tree *btree.BTree[*keyRev]) map[Revision]struct{} {
	revs := make(map[Revision]struct{}, tree.Len())
	tree.Ascend(func(kr *keyRev) bool {
		revs[kr.mod] = struct{}{}
		return true
	})
	return revs
}

// at returns a copy of the snapshot of rev, or false if rev was compacted. It
// panics if rev is newer than the latest revision.
func (ti *treeIndex) at(rev int64) (snapshot, bool) {
	ti.RLock()
	defer ti.RUnlock()
	if sn, ok := ti.snapshots.At(rev); ok {
		return *sn, true
	}
	if latest := ti.snapshots.Latest().rev; rev > latest {
		ti.lg.Panic("revision is newer than the latest revision in the index",
			zap.Int64("latest-revision", latest), zap.Int64("revision", rev))
	}
	return snapshot{}, false
}

// mustAt is at for Compact and Keep, which panic if rev was compacted.
func (ti *treeIndex) mustAt(rev int64) snapshot {
	sn, ok := ti.at(rev)
	if !ok {
		ti.lg.Panic("revision has been compacted in the index", zap.Int64("revision", rev))
	}
	return sn
}
