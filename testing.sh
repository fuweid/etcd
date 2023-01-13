#!/usr/bin/env bash

set -euo pipefail

while true
do
PASSES='build grpcproxy' CPU='4' RACE='true' ./test
done
