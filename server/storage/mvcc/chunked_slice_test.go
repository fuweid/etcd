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
	"math"
	"runtime"
	"testing"
	"weak"

	"github.com/stretchr/testify/require"
)

func TestChunkedSliceAppend(t *testing.T) {
	var s chunkedSlice[int]
	require.Zero(t, s.Len())

	var ptrs []*int
	for i := range 2*chunkLen + 1 {
		p := s.Append(i)
		ptrs = append(ptrs, p)
		require.Equal(t, i, *p)
		require.Equal(t, i+1, s.Len())
		require.Same(t, p, s.Back())
	}

	require.Len(t, s.chunks, 3)
	require.Same(t, ptrs[0], s.Front())
	for i, p := range ptrs {
		require.Same(t, p, s.At(i))
		require.Equal(t, i, *p)
	}
}

func TestChunkedSliceDropFront(t *testing.T) {
	tests := []struct {
		name       string
		size, drop int
		wantChunks int
	}{
		{name: "empty", wantChunks: 0},
		{name: "drop nothing", size: 2 * chunkLen, wantChunks: 2},
		{name: "drop within the first chunk", size: 2 * chunkLen, drop: 10, wantChunks: 2},
		{name: "drop the first chunk", size: 2 * chunkLen, drop: chunkLen, wantChunks: 1},
		{name: "drop into the second chunk", size: 2 * chunkLen, drop: chunkLen + 10, wantChunks: 1},
		{name: "drop several chunks", size: 4 * chunkLen, drop: 3*chunkLen + 10, wantChunks: 1},
		{name: "drop all but one", size: 2 * chunkLen, drop: 2*chunkLen - 1, wantChunks: 1},
		{name: "drop all at a chunk boundary", size: 2 * chunkLen, drop: 2 * chunkLen, wantChunks: 0},
		{name: "drop all within a chunk", size: chunkLen + 3, drop: chunkLen + 3, wantChunks: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var s chunkedSlice[int]
			for i := range tt.size {
				s.Append(i)
			}

			s.DropFront(tt.drop)
			require.Equal(t, tt.size-tt.drop, s.Len())
			require.Len(t, s.chunks, tt.wantChunks)
			for i := range s.Len() {
				require.Equal(t, tt.drop+i, *s.At(i))
			}
			if len(s.chunks) > 0 {
				for _, v := range s.chunks[0][:s.head] {
					require.Zero(t, v)
				}
			}

			// Make sure Append still works after DropFront.
			s.Append(tt.size)
			require.Equal(t, tt.size-tt.drop+1, s.Len())
			require.Equal(t, tt.drop, *s.Front())
			require.Equal(t, tt.size, *s.Back())
		})
	}
}

func TestChunkedSliceOutOfRange(t *testing.T) {
	tests := []struct {
		name      string
		add, drop int
	}{
		{name: "empty"},
		{name: "dropped and free slots", add: chunkLen + 3, drop: 2},
		{name: "drop all within a chunk", add: chunkLen + 3, drop: chunkLen + 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var s chunkedSlice[int]
			for i := range tt.add {
				s.Append(i)
			}
			s.DropFront(tt.drop)
			n := tt.add - tt.drop

			require.Panicsf(t, func() { s.At(-1) }, "at(-1)")
			require.Panicsf(t, func() { s.At(n) }, "at(%d)", n)
			require.Panicsf(t, func() { s.DropFront(-1) }, "dropFront(-1)")
			require.Panicsf(t, func() { s.DropFront(n + 1) }, "dropFront(%d)", n+1)
			if n == 0 {
				require.Panicsf(t, func() { s.Front() }, "front")
				require.Panicsf(t, func() { s.Back() }, "back")
			}
			require.Equal(t, n, s.Len())
			for i := range n {
				require.Equal(t, tt.drop+i, *s.At(i))
			}
		})
	}
}

func TestChunkedSliceDropFrontReleasesMemory(t *testing.T) {
	// NOTE:
	//
	// The weak.Pointer documentation notes that batching typically
	// only happens for tiny (on the order of 16 bytes or less), pointer-free
	// objects. Use 64 bytes to avoid sharing an allocation with an object
	// that stays alive.
	//
	// REF: https://pkg.go.dev/weak#Pointer
	type elem [64]byte
	var s chunkedSlice[*elem]
	for range 2 * chunkLen {
		s.Append(new(elem))
	}

	const drop = chunkLen + 1
	droppedChunk := weak.Make(s.chunks[0])
	inDroppedChunk := weak.Make(*s.At(0))
	inKeptChunk := weak.Make(*s.At(drop - 1))
	kept := weak.Make(*s.At(drop))

	s.DropFront(drop)
	runtime.GC()

	require.Nilf(t, droppedChunk.Value(), "the dropped chunk is still referenced")
	require.Nilf(t, inDroppedChunk.Value(), "the element in the dropped chunk is still referenced")
	require.Nilf(t, inKeptChunk.Value(), "the element dropped from the kept chunk is still referenced")
	require.NotNilf(t, kept.Value(), "the first element that is kept")
	runtime.KeepAlive(&s)
}

func FuzzChunkedSlice(f *testing.F) {
	// Each input appends add elements, then drops up to drop elements.
	f.Add(uint16(0), uint16(0))
	f.Add(uint16(0), uint16(1))
	f.Add(uint16(1), uint16(0))
	f.Add(uint16(3), uint16(1))
	f.Add(uint16(chunkLen), uint16(chunkLen))
	f.Add(uint16(3*chunkLen), uint16(2*chunkLen+10))
	f.Add(uint16(0), uint16(math.MaxUint16))
	f.Add(uint16(math.MaxUint16), uint16(0))

	// NOTE:
	//
	// All the inputs use one chunkedSlice, so that an input starts with
	// what the previous inputs left, instead of an empty chunkedSlice. A
	// failure depends on the previous inputs, so the failing input alone
	// may not reproduce it. The log has the layout that the input started
	// with.
	const maxSize = 16 * chunkLen
	var s chunkedSlice[int]
	var want []int
	var ptrs []*int
	next := 1

	check := func(t *testing.T) {
		t.Helper()
		require.Equal(t, len(want), s.Len())
		for i, v := range want {
			if p := s.At(i); p != ptrs[i] || *p != v {
				t.Fatalf("At(%d) = %d at %p, want %d at %p", i, *p, p, v, ptrs[i])
			}
		}
		if len(want) > 0 {
			require.Same(t, ptrs[0], s.Front())
			require.Same(t, ptrs[len(ptrs)-1], s.Back())
		}
	}

	f.Fuzz(func(t *testing.T, add, drop uint16) {
		t.Logf("start with head=%d, size=%d, chunks=%d", s.head, s.size, len(s.chunks))

		for range add {
			ptrs = append(ptrs, s.Append(next))
			want = append(want, next)
			next++
		}
		check(t)

		// Drop the elements over maxSize as well, to bound the cost of check.
		n := min(max(int(drop), len(want)-maxSize), len(want))
		s.DropFront(n)
		want, ptrs = want[n:], ptrs[n:]
		check(t)
	})
}
