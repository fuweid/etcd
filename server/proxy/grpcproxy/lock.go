// Copyright 2017 The etcd Lockors
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

package grpcproxy

import (
	"context"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3lock/v3lockpb"
)

type lockProxy struct {
	lockClient v3lockpb.LockClient

	v3lockpb.UnimplementedLockServer
}

func NewLockProxy(client *clientv3.Client) v3lockpb.LockServer {
	return &lockProxy{lockClient: v3lockpb.NewLockClient(client.ActiveConnection())}
}

func (lp *lockProxy) Lock(ctx context.Context, req *v3lockpb.LockRequest) (*v3lockpb.LockResponse, error) {
	return lp.lockClient.Lock(ctx, req)
}

func (lp *lockProxy) Unlock(ctx context.Context, req *v3lockpb.UnlockRequest) (*v3lockpb.UnlockResponse, error) {
	return lp.lockClient.Unlock(ctx, req)
}
