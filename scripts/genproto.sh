#!/usr/bin/env bash
#
# Generate all etcd protobuf bindings.
# Run from repository root directory named etcd.
#
set -euo pipefail

shopt -s globstar

if ! [[ "$0" =~ scripts/genproto.sh ]]; then
  echo "must be run from repository root"
  exit 255
fi

source ./scripts/test_lib.sh

if [[ $(protoc --version | cut -f2 -d' ') != "3.14.0" ]]; then
  echo "could not find protoc 3.14.0, is it installed + in PATH?"
  exit 255
fi

readonly googleapi_commit=77c99e43177c76ae1c1edacee7b6ac4e35a42f3d

GOFAST_BIN=$(tool_get_bin github.com/gogo/protobuf/protoc-gen-gofast)
PROTOBUFGO_BIN=$(tool_get_bin google.golang.org/protobuf/cmd/protoc-gen-go)
GRPC_GATEWAY_BIN=$(tool_get_bin github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway)
OPENAPIV2_BIN=$(tool_get_bin github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2)
VTPROTOBUFGO_BIN=$(tool_get_bin github.com/planetscale/vtprotobuf/cmd/protoc-gen-go-vtproto)
GOGOPROTO_ROOT="$(tool_pkg_dir github.com/gogo/protobuf/proto)/.."
GRPC_GATEWAY_ROOT="$(tool_pkg_dir github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway)/.."
RAFT_ROOT="$(tool_pkg_dir go.etcd.io/raft/v3/raftpb)/.."
GOOGLEAPI_ROOT=$(mktemp -d -t 'googleapi.XXXXX')

trap "rm -rf ${GOOGLEAPI_ROOT}" EXIT

function download_googleapi() {
  run pushd $GOOGLEAPI_ROOT
  run git init
  run git remote add upstream https://github.com/googleapis/googleapis.git
  run git fetch upstream $googleapi_commit
  run git reset --hard FETCH_HEAD
  run popd
}

download_googleapi

echo
echo "Resolved binary and packages versions:"
echo "  - protoc-gen-gofast:       ${GOFAST_BIN}"
echo "  - protoc-gen-grpc-gateway: ${GRPC_GATEWAY_BIN}"
echo "  - openapiv2:               ${OPENAPIV2_BIN}"
echo "  - gogoproto-root:          ${GOGOPROTO_ROOT}"
echo "  - grpc-gateway-root:       ${GRPC_GATEWAY_ROOT}"
echo "  - raft-root:               ${RAFT_ROOT}"
GOGOPROTO_PATH="${GOGOPROTO_ROOT}:${GOGOPROTO_ROOT}/protobuf"

# directories containing protos to be built
DIRS="./api/etcdserverpb ./server/etcdserver/api/snap/snappb ./api/mvccpb ./server/lease/leasepb ./api/authpb ./server/etcdserver/api/v3lock/v3lockpb ./server/etcdserver/api/v3election/v3electionpb ./api/membershippb ./api/versionpb"

log_callout -e "\\nRunning gofast (gogo) proto generation..."

for dir in ${DIRS}; do
  run pushd "${dir}"
    run protoc --experimental_allow_proto3_optional \
      -I=".:${ETCD_ROOT_DIR}/..:${RAFT_ROOT}:${ETCD_ROOT_DIR}:${GOOGLEAPI_ROOT}" \
      -I"${GRPC_GATEWAY_ROOT}" \
      --plugin="${PROTOBUFGO_BIN}" \
      --plugin="${VTPROTOBUFGO_BIN}" \
      --go-grpc_out=paths=source_relative:. \
      --go_out=paths=source_relative:. \
      --go-vtproto_out=paths=source_relative:. \
      --go-vtproto_opt=features=marshal+unmarshal+size \
      ./**/*.proto

    #if [[ "${dir}" == "./api/etcdserverpb" ]]; then
    #  run protoc -I=".:${ETCD_ROOT_DIR}/..:${RAFT_ROOT}:${ETCD_ROOT_DIR}:${GOOGLEAPI_ROOT}" \
    #    -I"${GRPC_GATEWAY_ROOT}" \
    #    --gotag_out=auto="json",paths=source_relative:. ./rpc.proto
    #fi

    run sed -i -E "s|SizeVT|Size|g" ./**/*_vtproto.pb.go || echo
    run sed -i -E "s|MarshalVT|Marshal|g" ./**/*_vtproto.pb.go || echo 
    run sed -i -E "s|UnmarshalVT|Unmarshal|g" ./**/*_vtproto.pb.go || echo

    run gofmt -s -w ./**/*.pb.go
    run_go_tool "golang.org/x/tools/cmd/goimports" -w ./**/*.pb.go
  run popd
done

# still use gogo because of raftpb
run pushd "./server/storage/wal/walpb"
  run protoc \
    -I=".:$GOGOPROTO_PATH:${ETCD_ROOT_DIR}/..:${RAFT_ROOT}:${ETCD_ROOT_DIR}:${GOOGLEAPI_ROOT}" \
    -I"${GRPC_GATEWAY_ROOT}" \
    --plugin="${GOFAST_BIN}" \
    --gofast_out=paths=source_relative,Mraftpb/raft.proto=go.etcd.io/raft/v3/raftpb,Mgoogle/protobuf/descriptor.proto=github.com/gogo/protobuf/protoc-gen-gogo/descriptor:. \
    ./**/*.proto

  run gofmt -s -w ./**/*.pb.go
  run_go_tool "golang.org/x/tools/cmd/goimports" -w ./**/*.pb.go
run popd

log_callout -e "\\nRunning swagger & grpc_gateway proto generation..."

# remove old swagger files so it's obvious whether the files fail to generate
rm -rf Documentation/dev-guide/apispec/swagger/*json
for pb in api/etcdserverpb/rpc server/etcdserver/api/v3lock/v3lockpb/v3lock server/etcdserver/api/v3election/v3electionpb/v3election; do
  log_callout "grpc & swagger for: ${pb}.proto"
  run protoc --experimental_allow_proto3_optional \
      -I. \
      -I"${GOOGLEAPI_ROOT}" \
      -I"${GRPC_GATEWAY_ROOT}" \
      -I"${GOGOPROTO_PATH}" \
      -I"${ETCD_ROOT_DIR}/.." \
      -I"${RAFT_ROOT}" \
      --grpc-gateway_out=logtostderr=true,Mgoogle/protobuf/descriptor.proto=github.com/gogo/protobuf/protoc-gen-gogo/descriptor,Mgoogle/protobuf/struct.proto=github.com/gogo/protobuf/types,paths=source_relative,standalone=true:./ \
      --openapiv2_out=json_names_for_fields=false,logtostderr=true:./Documentation/dev-guide/apispec/swagger/. \
      --openapiv2_opt=Mgoogle/protobuf/descriptor.proto=github.com/gogo/protobuf/protoc-gen-gogo/descriptor,Mgoogle/protobuf/struct.proto=github.com/gogo/protobuf/types:. \
      --plugin="${OPENAPIV2_BIN}" \
      --plugin="${GRPC_GATEWAY_BIN}" \
      ${pb}.proto

  # hack to move gw files around so client won't include them
  pkgpath=$(dirname "${pb}")
  pkg=$(basename "${pkgpath}")
  gwfile="${pb}.pb.gw.go"

  run sed -i -E "s#package $pkg#package gw#g" "${gwfile}"
  run go fmt "${gwfile}"

  gwdir="${pkgpath}/gw/"
  run mkdir -p "${gwdir}"
  run mv "${gwfile}" "${gwdir}"

  swaggerName=$(basename ${pb})
  run mv  Documentation/dev-guide/apispec/swagger/${pb}.swagger.json \
    Documentation/dev-guide/apispec/swagger/"${swaggerName}".swagger.json
done

if [ "${1:-}" != "--skip-protodoc" ]; then
  log_callout "protodoc is auto-generating grpc API reference documentation..."

  # API reference
  API_REFERENCE_FILE="Documentation/dev-guide/api_reference_v3.md"
  run rm -rf ${API_REFERENCE_FILE}
  run_go_tool go.etcd.io/protodoc --directories="api/etcdserverpb=service_message,api/mvccpb=service_message,server/lease/leasepb=service_message,api/authpb=service_message" \
    --output="${API_REFERENCE_FILE}" \
    --message-only-from-this-file="api/etcdserverpb/rpc.proto" \
    --disclaimer="---
title: API reference
---

This API reference is autogenerated from the named \`.proto\` files." || exit 2

  # remove the first 3 lines of the doc as an empty --title adds '### ' to the top of the file.
  run sed -i -e 1,3d ${API_REFERENCE_FILE}

  # API reference: concurrency
  API_REFERENCE_CONCURRENCY_FILE="Documentation/dev-guide/api_concurrency_reference_v3.md"
  run rm -rf ${API_REFERENCE_CONCURRENCY_FILE}
  run_go_tool go.etcd.io/protodoc --directories="server/etcdserver/api/v3lock/v3lockpb=service_message,server/etcdserver/api/v3election/v3electionpb=service_message,api/mvccpb=service_message" \
    --output="${API_REFERENCE_CONCURRENCY_FILE}" \
    --disclaimer="---
title: \"API reference: concurrency\"
---

This API reference is autogenerated from the named \`.proto\` files." || exit 2

  # remove the first 3 lines of the doc as an empty --title adds '### ' to the top of the file.
  run sed -i -e 1,3d ${API_REFERENCE_CONCURRENCY_FILE}

  log_success "protodoc is finished."
  log_warning -e "\\nThe API references have NOT been automatically published on the website."
  log_success -e "\\nTo publish the API references, copy the following files"
  log_success "  - ${API_REFERENCE_FILE}"
  log_success "  - ${API_REFERENCE_CONCURRENCY_FILE}"
  log_success "to the etcd-io/website repo under the /content/en/docs/next/dev-guide/ folder."
  log_success "(https://github.com/etcd-io/website/tree/main/content/en/docs/next/dev-guide)"
else
  log_warning "skipping grpc API reference document auto-generation..."
fi

log_success -e "\\n./genproto SUCCESS"
