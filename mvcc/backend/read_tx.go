// Copyright 2017 The etcd Authors
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

package backend

import (
	"math"
	"sync"

	bolt "go.etcd.io/bbolt"
	"go.etcd.io/etcd/mvcc/buckets"
)

// IsSafeRangeBucket is a hack to avoid inadvertently reading duplicate keys;
// overwrites on a bucket should only fetch with limit=1, but IsSafeRangeBucket
// is known to never overwrite any key so range is safe.

type ReadTx interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()

	UnsafeRange(bucket buckets.Bucket, key, endKey []byte, limit int64) (keys [][]byte, vals [][]byte)
	UnsafeForEach(bucket buckets.Bucket, visitor func(k, v []byte) error) error
}

type readTx struct {
	// mu protects accesses to the txReadBuffer
	mu  sync.RWMutex
	buf txReadBuffer

	// TODO: group and encapsulate {txMu, tx, buckets, txWg}, as they share the same lifecycle.
	// txMu protects accesses to buckets and tx on Range requests.
	txMu    sync.RWMutex
	tx      *bolt.Tx
	buckets map[buckets.BucketID]*bolt.Bucket
	// txWg protects tx from being rolled back at the end of a batch interval until all reads using this tx are done.
	txWg *sync.WaitGroup
}

func (rt *readTx) Lock()    { rt.mu.Lock() }
func (rt *readTx) Unlock()  { rt.mu.Unlock() }
func (rt *readTx) RLock()   { rt.mu.RLock() }
func (rt *readTx) RUnlock() { rt.mu.RUnlock() }

func (rt *readTx) UnsafeRange(bucketType buckets.Bucket, key, endKey []byte, limit int64) ([][]byte, [][]byte) {
	if endKey == nil {
		// forbid duplicates for single keys
		limit = 1
	}
	if limit <= 0 {
		limit = math.MaxInt64
	}
	if limit > 1 && !bucketType.IsSafeRangeBucket() {
		panic("do not use unsafeRange on non-keys bucket")
	}
	keys, vals := rt.buf.Range(bucketType, key, endKey, limit)
	if int64(len(keys)) == limit {
		return keys, vals
	}

	// find/cache bucket
	bn := bucketType.ID()
	rt.txMu.RLock()
	bucket, ok := rt.buckets[bn]
	rt.txMu.RUnlock()
	if !ok {
		rt.txMu.Lock()
		bucket = rt.tx.Bucket(bucketType.Name())
		rt.buckets[bn] = bucket
		rt.txMu.Unlock()
	}

	// ignore missing bucket since may have been created in this batch
	if bucket == nil {
		return keys, vals
	}
	rt.txMu.Lock()
	c := bucket.Cursor()
	rt.txMu.Unlock()

	k2, v2 := unsafeRange(c, key, endKey, limit-int64(len(keys)))
	return append(k2, keys...), append(v2, vals...)
}

func (rt *readTx) UnsafeForEach(bucket buckets.Bucket, visitor func(k, v []byte) error) error {
	dups := make(map[string]struct{})
	getDups := func(k, v []byte) error {
		dups[string(k)] = struct{}{}
		return nil
	}
	visitNoDup := func(k, v []byte) error {
		if _, ok := dups[string(k)]; ok {
			return nil
		}
		return visitor(k, v)
	}
	if err := rt.buf.ForEach(bucket, getDups); err != nil {
		return err
	}
	rt.txMu.Lock()
	err := unsafeForEach(rt.tx, bucket, visitNoDup)
	rt.txMu.Unlock()
	if err != nil {
		return err
	}
	return rt.buf.ForEach(bucket, visitor)
}

func (rt *readTx) reset() {
	rt.buf.reset()
	rt.buckets = make(map[buckets.BucketID]*bolt.Bucket)
	rt.tx = nil
	rt.txWg = new(sync.WaitGroup)
}

// TODO: create a base type for readTx and concurrentReadTx to avoid duplicated function implementation?
type concurrentReadTx struct {
	buf     txReadBuffer
	txMu    *sync.RWMutex
	tx      *bolt.Tx
	buckets map[buckets.BucketID]*bolt.Bucket
	txWg    *sync.WaitGroup
}

func (rt *concurrentReadTx) Lock()   {}
func (rt *concurrentReadTx) Unlock() {}

// RLock is no-op. concurrentReadTx does not need to be locked after it is created.
func (rt *concurrentReadTx) RLock() {}

// RUnlock signals the end of concurrentReadTx.
func (rt *concurrentReadTx) RUnlock() { rt.txWg.Done() }

func (rt *concurrentReadTx) UnsafeForEach(bucket buckets.Bucket, visitor func(k, v []byte) error) error {
	dups := make(map[string]struct{})
	getDups := func(k, v []byte) error {
		dups[string(k)] = struct{}{}
		return nil
	}
	visitNoDup := func(k, v []byte) error {
		if _, ok := dups[string(k)]; ok {
			return nil
		}
		return visitor(k, v)
	}
	if err := rt.buf.ForEach(bucket, getDups); err != nil {
		return err
	}
	rt.txMu.Lock()
	err := unsafeForEach(rt.tx, bucket, visitNoDup)
	rt.txMu.Unlock()
	if err != nil {
		return err
	}
	return rt.buf.ForEach(bucket, visitor)
}

func (rt *concurrentReadTx) UnsafeRange(bucketType buckets.Bucket, key, endKey []byte, limit int64) ([][]byte, [][]byte) {
	if endKey == nil {
		// forbid duplicates for single keys
		limit = 1
	}
	if limit <= 0 {
		limit = math.MaxInt64
	}
	if limit > 1 && !bucketType.IsSafeRangeBucket() {
		panic("do not use unsafeRange on non-keys bucket")
	}
	keys, vals := rt.buf.Range(bucketType, key, endKey, limit)
	if int64(len(keys)) == limit {
		return keys, vals
	}

	// find/cache bucket
	bn := bucketType.ID()
	rt.txMu.RLock()
	bucket, ok := rt.buckets[bn]
	rt.txMu.RUnlock()
	if !ok {
		rt.txMu.Lock()
		bucket = rt.tx.Bucket(bucketType.Name())
		rt.buckets[bn] = bucket
		rt.txMu.Unlock()
	}

	// ignore missing bucket since may have been created in this batch
	if bucket == nil {
		return keys, vals
	}
	rt.txMu.Lock()
	c := bucket.Cursor()
	rt.txMu.Unlock()

	k2, v2 := unsafeRange(c, key, endKey, limit-int64(len(keys)))
	return append(k2, keys...), append(v2, vals...)
}
