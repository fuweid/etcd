// Copyright 2026 The etcd Authors
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
	"k8s.io/utils/third_party/forked/golang/btree"

	"go.etcd.io/etcd/client/pkg/v3/verify"
)

// snapshot is the index state for a main revision. Multiple changes in one txn
// will share the same snapshot. The index state will remove tombstoned keys
// from the tree. However, we need to keep the tombstoned revisions for the
// purpose of compaction.
type snapshot struct {
	rev int64

	tree *btree.BTree[*keyRev]

	tombstones []Revision
}

// snapshots is the history of the index, oldest first. It has a snapshot for
// every revision from the compacted one to the latest one, so the snapshot of
// a revision is at its distance from the oldest one.
//
// For example, with the index compacted at 5, and the latest revision 8:
//
//	history   [5]  [6]  [7]  [8]
//
//	At(4)     compacted
//	At(7)               [7]
//	At(9)     not written yet
//
// A *snapshot stays valid when snapshots are added, but not after Compact. It
// does no locking.
type snapshots struct {
	history chunkedSlice[snapshot]
}

func (s *snapshots) Len() int { return s.history.Len() }

func (s *snapshots) Oldest() *snapshot { return s.history.Front() }

func (s *snapshots) Latest() *snapshot { return s.history.Back() }

// At returns the snapshot of rev, or false if the history does not have rev:
// it was compacted, or it is not written yet.
func (s *snapshots) At(rev int64) (*snapshot, bool) {
	i := rev - s.Oldest().rev
	if i < 0 || i >= int64(s.Len()) {
		return nil, false
	}
	return s.history.At(int(i)), true
}

// Append adds sn, which must be at the revision after the latest snapshot.
func (s *snapshots) Append(sn snapshot) *snapshot {
	verify.Verify("snapshot should be at the revision after the latest snapshot", func() (bool, map[string]any) {
		return s.Len() == 0 || sn.rev == s.Latest().rev+1, map[string]any{"revision": sn.rev}
	})
	return s.history.Append(sn)
}

// Compact drops the snapshots before rev, which must be in the history.
//
//	             [5]  [6]  [7]  [8]
//	Compact(7)             [7]  [8]
func (s *snapshots) Compact(rev int64) {
	s.history.DropFront(int(rev - s.Oldest().rev))
}
