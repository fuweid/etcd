// Copyright 2021 The etcd Authors
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

package buckets

import (
	"bytes"
)

type BucketID int

type Bucket interface {
	// ID returns a unique identifier of a bucket.
	// The id must NOT be persisted and can be used as lightweight identificator
	// in the in-memory maps.
	ID() BucketID
	Name() []byte
	// String implements Stringer (human readable name).
	String() string

	// IsSafeRangeBucket is a hack to avoid inadvertently reading duplicate keys;
	// overwrites on a bucket should only fetch with limit=1, but safeRangeBucket
	// is known to never overwrite any key so range is safe.
	IsSafeRangeBucket() bool
}

var (
	keyBucketName   = []byte("key")
	metaBucketName  = []byte("meta")
	leaseBucketName = []byte("lease")
	alarmBucketName = []byte("alarm")

	clusterBucketName = []byte("cluster")

	membersBucketName        = []byte("members")
	membersRemovedBucketName = []byte("members_removed")

	authBucketName      = []byte("auth")
	authUsersBucketName = []byte("authUsers")
	authRolesBucketName = []byte("authRoles")

	testBucketName = []byte("test")
)

var (
	Key     = Bucket(bucket{id: 1, name: keyBucketName, safeRangeBucket: true})
	Meta    = Bucket(bucket{id: 2, name: metaBucketName, safeRangeBucket: false})
	Lease   = Bucket(bucket{id: 3, name: leaseBucketName, safeRangeBucket: false})
	Alarm   = Bucket(bucket{id: 4, name: alarmBucketName, safeRangeBucket: false})
	Cluster = Bucket(bucket{id: 5, name: clusterBucketName, safeRangeBucket: false})

	Members        = Bucket(bucket{id: 10, name: membersBucketName, safeRangeBucket: false})
	MembersRemoved = Bucket(bucket{id: 11, name: membersRemovedBucketName, safeRangeBucket: false})

	Auth      = Bucket(bucket{id: 20, name: authBucketName, safeRangeBucket: false})
	AuthUsers = Bucket(bucket{id: 21, name: authUsersBucketName, safeRangeBucket: false})
	AuthRoles = Bucket(bucket{id: 22, name: authRolesBucketName, safeRangeBucket: false})

	Test = Bucket(bucket{id: 100, name: testBucketName, safeRangeBucket: false})
)

type bucket struct {
	id              BucketID
	name            []byte
	safeRangeBucket bool
}

func (b bucket) ID() BucketID            { return b.id }
func (b bucket) Name() []byte            { return b.name }
func (b bucket) String() string          { return string(b.Name()) }
func (b bucket) IsSafeRangeBucket() bool { return b.safeRangeBucket }

var (
	MetaConsistentIndexKeyName = []byte("consistent_index")
	MetaTermKeyName            = []byte("term")
)

// DefaultIgnores defines buckets & keys to ignore in hash checking.
func DefaultIgnores(bucket, key []byte) bool {
	// consistent index & term might be changed due to v2 internal sync, which
	// is not controllable by the user.
	return bytes.Compare(bucket, Meta.Name()) == 0 &&
		(bytes.Compare(key, MetaTermKeyName) == 0 || bytes.Compare(key, MetaConsistentIndexKeyName) == 0)
}
