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
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/third_party/forked/golang/btree"
)

// newSnapshots returns the snapshots of the revisions from oldest to latest.
// Every snapshot has its own tree, to tell them apart.
func newSnapshots(oldest, latest int64) *snapshots {
	s := &snapshots{}
	for rev := oldest; rev <= latest; rev++ {
		s.Append(snapshot{rev: rev, tree: btree.New(2, keyRevLess)})
	}
	return s
}

func TestSnapshotsAt(t *testing.T) {
	// More revisions than a chunk of the history has.
	oldest, latest := int64(5), int64(5+3*chunkLen)
	s := newSnapshots(oldest, latest)

	require.Equal(t, int(latest-oldest)+1, s.Len())
	require.Equal(t, oldest, s.Oldest().rev)
	require.Equal(t, latest, s.Latest().rev)

	for rev := oldest; rev <= latest; rev++ {
		sn, ok := s.At(rev)
		require.Truef(t, ok, "at(%d)", rev)
		require.Equalf(t, rev, sn.rev, "at(%d)", rev)
	}
	_, ok := s.At(oldest - 1)
	require.Falsef(t, ok, "the revision before the oldest snapshot")
	_, ok = s.At(latest + 1)
	require.Falsef(t, ok, "the revision after the latest snapshot")
}

func TestSnapshotsCompact(t *testing.T) {
	tests := []struct {
		name string
		rev  int64
		wlen int
	}{
		{name: "at the oldest", rev: 5, wlen: 4},
		{name: "in the middle", rev: 7, wlen: 2},
		{name: "at the latest", rev: 8, wlen: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newSnapshots(5, 8)
			kept, _ := s.At(tt.rev)
			tree := kept.tree

			s.Compact(tt.rev)

			require.Equal(t, tt.wlen, s.Len())
			require.Equal(t, tt.rev, s.Oldest().rev)
			require.Same(t, tree, s.Oldest().tree)
			require.Equal(t, int64(8), s.Latest().rev)
			_, ok := s.At(tt.rev - 1)
			require.Falsef(t, ok, "the revision before the compacted one")
		})
	}
}
