#!/usr/bin/env bash
# Copyright 2025 The etcd Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source ./scripts/test_lib.sh

GO_CMD="go"
fuzz_time=${FUZZ_TIME:-"300s"}
target_path=${TARGET_PATH:-"./server/etcdserver/api/v3rpc"}

function run_fuzz {
    local package_path=$1
    local target=$2

    log_callout "Fuzzing ${target} in ${package_path} for ${fuzz_time}"
    (
        cd "${package_path}"
        "${GO_CMD}" test -fuzz "^${target}$" -fuzztime "${fuzz_time}"
    )
    log_success "Completed fuzzing ${target} in ${package_path}"
}

for target in FuzzTxnRangeRequest FuzzTxnPutRequest FuzzTxnDeleteRangeRequest; do
    run_fuzz "${target_path}" "${target}"
done

run_fuzz "./server/storage/mvcc" FuzzChunkedSlice
