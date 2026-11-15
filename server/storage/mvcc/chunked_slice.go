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
	"fmt"
	"slices"
)

// chunkLen is the number of elements in a chunk of chunkedSlice.
const chunkLen = 256

// chunkedSlice is a slice kept as a two-dimensional array: a list of fixed-size
// chunks. Append never moves an element, so a pointer to an element stays valid
// until DropFront drops that element, and DropFront frees whole chunks instead
// of copying the rest of the slice.
//
// For example, with 4 elements per chunk, 13 Appends give
//
//	chunks[0]    chunks[1]    chunks[2]    chunks[3]
//	e  e  e  e   e  e  e  e   e  e  e  e   e  .  .  .
//	^head=0, size=13
//
// and then DropFront(6) frees the first chunk and two slots of the next one:
//
//	chunks[0]    chunks[1]    chunks[2]
//	-  -  e  e   e  e  e  e   e  .  .  .
//	      ^head=2, size=7
//
// e is an element, - a dropped one, and . a free slot. The i-th element is in
// the slot head+i, counting the slots from chunks[0][0].
//
// It does no locking.
type chunkedSlice[T any] struct {
	chunks []*[chunkLen]T
	head   int // slot of the first element, in chunks[0]
	size   int
}

func (s *chunkedSlice[T]) Len() int { return s.size }

// At returns the i-th element from the front. It panics if i is out of range.
func (s *chunkedSlice[T]) At(i int) *T {
	if i < 0 || i >= s.size {
		// NOTE: A constant, because At is not inlined with a formatted
		// message.
		panic("chunkedSlice: index out of range")
	}
	slot := s.head + i
	return &s.chunks[slot/chunkLen][slot%chunkLen]
}

func (s *chunkedSlice[T]) Front() *T { return s.At(0) }

func (s *chunkedSlice[T]) Back() *T { return s.At(s.size - 1) }

// Append adds v to the back, and returns the element in the slice.
func (s *chunkedSlice[T]) Append(v T) *T {
	if s.head+s.size == len(s.chunks)*chunkLen { // no free slot
		s.chunks = append(s.chunks, new([chunkLen]T))
	}
	s.size++
	back := s.Back()
	*back = v
	return back
}

// DropFront drops the first n elements, and zeroes their slots so that the
// slice no longer references them. It panics if n is out of range.
func (s *chunkedSlice[T]) DropFront(n int) {
	if n < 0 || n > s.size {
		panic(fmt.Sprintf("chunkedSlice: DropFront(%d) out of range with length %d", n, s.size))
	}
	head := s.head + n
	// Delete zeroes the end of the list, so that it does not keep the chunks.
	s.chunks = slices.Delete(s.chunks, 0, head/chunkLen)
	s.head, s.size = head%chunkLen, s.size-n
	if s.head > 0 {
		clear(s.chunks[0][:s.head])
	}
}
